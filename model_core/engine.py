import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MemeBacktest

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.loader = CryptoDataLoader()
        self.loader.load_data()
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=ModelConfig.TRAIN_STEPS,
            eta_min=1e-5
        )
        
        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        lord_target_keywords = ["attention", "qk_norm"]
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=lord_target_keywords
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=lord_target_keywords
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.bt = MemeBacktest()
        
        self.best_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': []
        }
        self.patience_counter = 0
        self.patience = ModelConfig.EARLY_STOP_PATIENCE
        self.avg_reward_ema = None
        self.best_avg_reward = -float('inf')

    def train(self):
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['attention', 'qk_norm']")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            stack_sizes = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            entropies = []
            tokens_list = []

            for t in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                # Syntax-guided masking: ensure only valid tokens are sampled
                syntax_mask = self.vm.compute_syntax_mask(stack_sizes, t, ModelConfig.MAX_FORMULA_LEN)
                logits = logits.masked_fill(~syntax_mask, float('-inf'))
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                # Update stack sizes based on sampled actions
                self.vm._ensure_device(ModelConfig.DEVICE)
                stack_sizes = stack_sizes + self.vm._delta_d[action]
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            best_updated = False
            
            for i in range(bs):
                formula = seqs[i].tolist()
                
                res = self.vm.execute(formula, self.loader.feat_tensor)
                
                if res is None:
                    rewards[i] = -2.0
                    continue
                
                if res.std() < 1e-4:
                    rewards[i] = -1.0
                    continue
                
                score, ret_val = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret)
                if torch.isnan(score) or torch.isinf(score):
                    rewards[i] = -2.0
                    continue
                rewards[i] = score
                
                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    best_updated = True
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}")
            
            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            loss = 0
            entropy_bonus = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
                entropy_bonus += entropies[t]

            # Linearly decay entropy coefficient from start to end value.
            progress = step / max(ModelConfig.TRAIN_STEPS - 1, 1)
            entropy_coeff = (
                ModelConfig.ENTROPY_COEFF_START
                + (ModelConfig.ENTROPY_COEFF_END - ModelConfig.ENTROPY_COEFF_START) * progress
            )
            loss = loss.mean() - entropy_coeff * entropy_bonus.mean()
            
            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=ModelConfig.GRAD_CLIP_NORM)
            self.opt.step()
            self.scheduler.step()
            
            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()

            # Logging
            avg_reward = rewards.mean().item()

            # Early stopping based on avg_reward EMA (not best_score)
            if self.avg_reward_ema is None:
                self.avg_reward_ema = avg_reward
            else:
                self.avg_reward_ema = 0.95 * self.avg_reward_ema + 0.05 * avg_reward

            if self.avg_reward_ema > self.best_avg_reward:
                self.best_avg_reward = self.avg_reward_ema
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            
            pbar.set_postfix(postfix_dict)

            if self.patience_counter >= self.patience:
                tqdm.write(
                    f"[!] Early stopping at step {step + 1}: avg_reward EMA "
                    f"({self.avg_reward_ema:.4f}) stagnated for {self.patience} steps."
                )
                break

        # Save best formula
        with open("best_meme_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        
        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
