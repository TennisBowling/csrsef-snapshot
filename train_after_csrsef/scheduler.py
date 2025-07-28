import random
import math
import logging
import threading
import multiprocessing
from typing import Dict, List, Tuple, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class AsyncHyperBandScheduler:
    def __init__(
        self,
        config_space: Dict[str, Dict],
        max_epochs: int = 10,
        eta: int = 3,
        num_trials: int = 20,
        metric: str = "accuracy",
        include_default: bool = True,
        seed: int = None,
        max_concurrent_trials: int = 1,
        track_per_epoch: bool = False,
    ):
        self.config_space = config_space
        self.max_epochs = max_epochs
        self.eta = eta
        self.min_epochs = 1
        self.num_trials = num_trials
        self.metric = metric
        self.include_default = include_default
        self.max_concurrent_trials = max_concurrent_trials
        self.track_per_epoch = track_per_epoch
        
        if seed is not None:
            random.seed(seed)
        
        self.s_max = math.floor(math.log(self.max_epochs / self.min_epochs, self.eta))
        
        self.brackets = self._initialize_brackets()
        self.trials = {}  # trial_id -> trial info
        self.running_trials = set()  # Set of currently running trial IDs
        self.completed_trials = {}
        self.next_trial_id = 0
        self.best_config = None
        self.best_score = float('-inf')
        self.best_trial_id = None
        
        self.default_config = {}
        for param_name, param_config in self.config_space.items():
            if "default" in param_config:
                self.default_config[param_name] = param_config["default"]
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger("AsyncHyperBandScheduler")
        
    def _initialize_brackets(self):
        brackets = []
        for s in range(self.s_max + 1):
            n = math.ceil((self.s_max + 1) / (s + 1) * self.eta ** s)
            r = self.max_epochs * self.eta ** (-s)
            
            bracket = {
                's': s,
                'n': n,  # Initial number of configurations
                'r': r,  # Initial resources per config (epochs)
                'configs': [],  # (trial_id, config, current_epoch)
                'results': {},  # trial_id -> {epoch: score}
            }
            brackets.append(bracket)
        return brackets
    
    def _sample_configuration(self):
        config = {}
        for param_name, param_config in self.config_space.items():
            if "values" in param_config:
                # Discrete values
                config[param_name] = random.choice(param_config["values"])
            elif "min" in param_config and "max" in param_config:
                param_type = param_config.get("type", "float")
                min_val, max_val = param_config["min"], param_config["max"]
                
                if param_type == "int":
                    config[param_name] = random.randint(min_val, max_val)
                elif param_type == "log":
                    # Log scale for learning rates, etc.
                    log_min = math.log(min_val)
                    log_max = math.log(max_val)
                    config[param_name] = math.exp(random.uniform(log_min, log_max))
                else:  # float
                    config[param_name] = random.uniform(min_val, max_val)
            else:
                if "default" in param_config:
                    config[param_name] = param_config["default"]
        
        return config
    
    def _get_next_bracket(self):
        min_pending = min(bracket['pending'] for bracket in self.brackets)
        candidates = [i for i, bracket in enumerate(self.brackets) 
                     if bracket['pending'] == min_pending]
        return random.choice(candidates)
    
    def _promote_config(self, bracket_idx, trial_id, current_epoch):
        bracket = self.brackets[bracket_idx]
        s, n, r = bracket['s'], bracket['n'], bracket['r']
        
        configs_at_milestone = [
            (tid, bracket['results'][tid].get(current_epoch, float('-inf')))
            for tid in bracket['results']
            if current_epoch in bracket['results'][tid]
        ]
        
        # Not enough configs to make a decision yet
        target_configs = math.ceil(n * self.eta ** (-current_epoch // r))
        if len(configs_at_milestone) < target_configs:
            next_epoch = current_epoch + r
            return min(next_epoch, self.max_epochs)  # Continue to next milestone
        
        configs_at_milestone.sort(key=lambda x: x[1], reverse=True)
        
        k = math.ceil(len(configs_at_milestone) / self.eta)
        
        trial_ids_to_keep = [tid for tid, _ in configs_at_milestone[:k]]
        if trial_id in trial_ids_to_keep:
            next_epoch = current_epoch + r
            return min(next_epoch, self.max_epochs)  # Promote to next milestone
        else:
            return None  # Stop this configuration
    
    def get_next_trial(self):
        with self._lock:
            if self.next_trial_id >= self.num_trials or len(self.running_trials) >= self.max_concurrent_trials:
                return None, None
            
            # If it's the first trial and we should include the default config
            if self.next_trial_id == 0 and self.include_default and self.default_config:
                trial_id = self.next_trial_id
                self.next_trial_id += 1
                
                # Use the default config with appropriate epochs
                config = self.default_config.copy()
                config["num_epochs"] = self.max_epochs
                
                bracket_idx = 0
                bracket = self.brackets[bracket_idx]
                bracket['configs'].append((trial_id, config, self.max_epochs))
                bracket['pending'] += 1
                
                # Track this trial
                self.trials[trial_id] = {
                    'config': config,
                    'bracket_idx': bracket_idx,
                    'current_epoch': self.max_epochs,
                    'status': 'RUNNING',
                    'is_default': True,
                    'epoch_results': {}
                }
                
                self.running_trials.add(trial_id)
                return trial_id, config
            
            # Otherwise, sample a new configuration
            bracket_idx = self._get_next_bracket()
            bracket = self.brackets[bracket_idx]
            
            # If the bracket still has slots for new configurations
            if len(bracket['configs']) < bracket['n']:
                # Sample a new configuration
                trial_id = self.next_trial_id
                self.next_trial_id += 1
                config = self._sample_configuration()
                
                epochs = max(1, int(bracket['r']))  # Initial resource allocation (epochs)
                config["num_epochs"] = epochs
                
                bracket['configs'].append((trial_id, config, epochs))
                bracket['pending'] += 1
                
                # Track this trial
                self.trials[trial_id] = {
                    'config': config,
                    'bracket_idx': bracket_idx,
                    'current_epoch': epochs,
                    'status': 'RUNNING',
                    'is_default': False,
                    'epoch_results': {}
                }
                
                self.running_trials.add(trial_id)
                return trial_id, config
            
            # No more new configs for this bracket, try another
            return self.get_next_trial()
    
    def get_multiple_trials(self, count=None):
        with self._lock:
            if count is None:
                count = self.max_concurrent_trials
            
            available_slots = min(
                count,
                self.max_concurrent_trials - len(self.running_trials)
            )
            
            trials = []
            for _ in range(available_slots):
                trial_id, config = self.get_next_trial()
                if trial_id is None:
                    break
                trials.append((trial_id, config))
            
            return trials
    
    def update_epoch_result(self, trial_id, epoch, score):
        with self._lock:
            if not self.track_per_epoch:
                # If not tracking per epoch, just continue
                return True
                
            if trial_id not in self.trials:
                raise ValueError(f"Trial {trial_id} not found")
            
            trial = self.trials[trial_id]
            bracket_idx = trial['bracket_idx']
            bracket = self.brackets[bracket_idx]
            
            # Record the epoch result
            if trial_id not in bracket['results']:
                bracket['results'][trial_id] = {}
            bracket['results'][trial_id][epoch] = score
            
            trial['epoch_results'][epoch] = score
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = trial['config'].copy()
                self.best_trial_id = trial_id
                self.logger.info(f"New best: Trial {trial_id}, epoch {epoch} with score {score:.4f}")
            
            # For intermediate epochs, check if we should continue
            if epoch < trial['current_epoch']:
                # If this is a rung evaluation point (i.e., epoch is divisible by bracket's r)
                r = bracket['r']
                if epoch % r == 0 and epoch > 0:
                    # Determine if this configuration should continue
                    next_epoch = self._promote_config(bracket_idx, trial_id, epoch)
                    if next_epoch is None:
                        self.logger.info(f"Early stopping trial {trial_id} at epoch {epoch}")
                        return False
            
            return True
    
    def update_result(self, trial_id, score):
        with self._lock:
            if trial_id not in self.trials:
                raise ValueError(f"Trial {trial_id} not found")
            
            trial = self.trials[trial_id]
            bracket_idx = trial['bracket_idx']
            bracket = self.brackets[bracket_idx]
            epochs = trial['current_epoch']
            
            # Record the result
            if trial_id not in bracket['results']:
                bracket['results'][trial_id] = {}
            bracket['results'][trial_id][epochs] = score
            
            trial['epoch_results'][epochs] = score
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = trial['config'].copy()
                self.best_trial_id = trial_id
                self.logger.info(f"New best model: Trial {trial_id} with score {score:.4f}")
            
            # Mark trial as completed
            self.running_trials.remove(trial_id)
            self.completed_trials[trial_id] = trial
            bracket['pending'] -= 1
            trial['status'] = 'COMPLETED'
            trial['final_score'] = score
            
            # Decide whether to continue this configuration or start a new one
            next_epochs = self._promote_config(bracket_idx, trial_id, epochs)
            if next_epochs:
                new_trial_id = self.next_trial_id
                self.next_trial_id += 1
                
                # Copy the config with updated epochs
                new_config = trial['config'].copy()
                new_config['num_epochs'] = next_epochs
                
                bracket['configs'].append((new_trial_id, new_config, next_epochs))
                bracket['pending'] += 1
                
                # Track this trial
                self.trials[new_trial_id] = {
                    'config': new_config,
                    'bracket_idx': bracket_idx,
                    'current_epoch': next_epochs,
                    'status': 'RUNNING',
                    'is_default': trial.get('is_default', False),
                    'parent_id': trial_id,
                    'epoch_results': {}
                }
                
                self.running_trials.add(new_trial_id)
                return new_trial_id, new_config
            else:
                return self.get_next_trial()
    
    def get_best_result(self):
        with self._lock:
            return {
                'trial_id': self.best_trial_id,
                'config': self.best_config,
                'score': self.best_score
            }
    
    def get_status(self):
        with self._lock:
            return {
                'total_trials': self.next_trial_id,
                'running': len(self.running_trials),
                'completed': len(self.completed_trials),
                'best_score': self.best_score,
                'best_trial_id': self.best_trial_id
            }
    
    def is_finished(self):
        with self._lock:
            return len(self.running_trials) == 0 and self.next_trial_id >= self.num_trials


def run_trial(scheduler, train_fn, train_dataset, test_dataset, extractor_node_types, device_id, trial_id, config):
    
    if trial_id is None:
        return None
    
    if scheduler.track_per_epoch:
        def epoch_callback(epoch, metrics):
            # Determine which metric to use
            metric_name = scheduler.metric
            score = metrics.get(metric_name, 0.0)
            
            continue_training = scheduler.update_epoch_result(trial_id, epoch, score)
            return continue_training
    else:
        epoch_callback = None
    
    # Train the model
    logger = logging.getLogger(f"Trial-{trial_id}")
    logger.info(f"Starting training with config: {config}")
    
    # train_dataset, test_dataset, num_node_types, config, epoch_callback, device_id
    model, test_acc, test_f1 = train_fn(
        trial_id,
        train_dataset,
        test_dataset,
        extractor_node_types,
        config,
        epoch_callback,
        device_id
    )
    
    # Determine which metric to use
    score = test_acc if scheduler.metric == "accuracy" else test_f1
    
    next_trial = scheduler.update_result(trial_id, score)
    logger.info(f"Completed with score: {score:.4f}")
    
    return next_trial