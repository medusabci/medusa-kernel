import numpy as np
import os, json, pickle, datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import warnings


class Grinder:
    def __init__(self, **hyparams):
        # Make hyparams dictionary
        opt_params = dict()
        for key, value in hyparams.items():
            value = [value] if not isinstance(value, list) else value
            opt_params[key] = np.arange(len(value))
        
        # Make grid
        self.hyparams = hyparams
        self.opt_params = opt_params
        self.hyparams_grid = list(ParameterGrid(hyparams))
        self.opt_grid = list(ParameterGrid(opt_params))
    
    def get_hyparams(self, idx):
        return self.hyparams_grid[idx]


class Optimizer:

    def __init__(self, obj, func, grinder, args=None, previous_hist=None):
        
        # Supported values
        self.supported_obj = ('minimize', 'maximize')
        self.supported_approach = ('grid', 'random', 'bayesian')
        
        # Check errors
        if not obj in self.supported_obj:
            raise ValueError("Parameter 'obj' must be one of these values: " +
                             str(self.supported_obj))
        
        if (args is not None) and (type(args)!=dict):
                raise Exception("Argument 'args' must be None or type dict")
                
        if (previous_hist is not None) and (type(previous_hist)!=dict):
                raise Exception("Argument 'previous_hist' must be None or type "
                                "dict")
        
        # Initialize
        self.obj = obj
        self.func = func
        self.args = args
        self.grinder = grinder
        
        # Initialize history and the optim directory
        if previous_hist is None:
            # Initialize directory
            dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")        
            self.optim_dir = os.getcwd() + '/optimization/optim_' + \
                             dir_name + '\\'
            os.makedirs(self.optim_dir)
            
            # Instantiate useful variables
            self.history = dict()
            self.history["grid_idx"] = list()
            self.history["score"] = list()
            self.history["info"] = list()
            self.history["optim_dir"] = self.optim_dir
        else:
            warnings.warn("Loading a previous history only makes sense if "
                          "arguments 'obj', 'func', 'grinder', 'args' and have "
                          "not changed.")
            self.history = previous_hist
            self.optim_dir = self.history["optim_dir"]
            try:
                os.makedirs(self.history["optim_dir"])
            except FileExistsError as e:
                pass
        
        # Bayesian optimization variables
        self.bayes_feat = None
        self.bayes_model = None
        self.bayes_n_samp = None
        self.rbf_length_scale = None
        self.bayes_parameters = False
        
        # Save the grinder
        with open(self.optim_dir + 'grinder.pkl', 'wb') as file:
            pickle.dump(self.grinder, file)
    
    def __pick_params(self, approach):
        if approach == 'grid':
            idx = self.history["grid_idx"][-1]+1 \
                if len(self.history["grid_idx"]) > 0 else 0
        elif approach == 'random':
            indexes = np.arange(len(self.grinder.opt_grid))
            pickable = np.delete(indexes, self.history["grid_idx"])
            np.random.shuffle(pickable)
            idx = self.__pick_new_indexes_randomly(1, repetition=False)[0]
        elif approach == 'bayesian':
            # Pick new random samples
            samp = self.__pick_new_indexes_randomly(
                int(self.bayes_n_samp), repetition=False
            )
            feat = self.__get_bayes_feat(samp)
            mu, std = self.bayes_model.predict(feat, return_std=True)
            # Get best feat until now
            with warnings.catch_warnings():
                # ignore generated warnings
                warnings.simplefilter("ignore")
                # Update best model
                if self.obj == 'maximize':
                    best = np.max(self.bayes_model.predict(self.bayes_feat))
                else:
                    best = np.min(self.bayes_model.predict(self.bayes_feat))
            # Calculate the probability of improvement or expected improvement
            probs = norm.cdf((np.squeeze(mu) - best) / (std+1e-9))
            if self.obj == 'maximize':
                idx = samp[np.argmax(probs)]
            else:
                idx = samp[np.argmin(probs)]
        else:
            pass
        hyparams = self.grinder.get_hyparams(idx)
        # Return
        return int(idx), hyparams
    
    def __pick_new_indexes_randomly(self, n, repetition=False):
        indexes = np.arange(len(self.grinder.opt_grid))
        pickable = np.delete(indexes, self.history["grid_idx"]) \
            if repetition is False else indexes
        np.random.shuffle(pickable)
        return pickable[0:n]
    
    def __get_bayes_feat(self, indexes):
        data = np.empty((len(indexes), len(self.grinder.opt_grid[0])))
        i = 0
        for idx in indexes:
            data[i, :] = [value for key, value in
                          self.grinder.opt_grid[idx].items()]
            i += 1
        return data
    
    def set_bayes_opt_params(self, n_samp, rbf_length_scale=1):
        self.bayes_n_samp = n_samp
        self.rbf_length_scale = rbf_length_scale
        self.bayes_parameters = True
    
    def optimize(self, max_iter, approach, save_history=True):
        # Check errors
        if max_iter > len(self.grinder.opt_grid):
            raise ValueError("Parameter 'max_iter' must be <= than the grid "
                             "size:" + str(len(self.grinder.opt_grid)))
            
        if not approach in self.supported_approach:
            raise ValueError("Parameter 'approach' must be one of these "
                             "values: " + str(self.supported_obj))
            
        if approach == 'bayesian':
            if len(self.history["score"]) == 0:
                raise Exception("No previous history for bayesian "
                                "optimization. Initialize with a random "
                                "optimization")
            if not self.bayes_parameters:
                raise Exception("You must set the bayesian optimization "
                                "parameters first. Call function "
                                "set_bayes_opt_params.")
                
        # Initialize bayesian model
        if approach == 'bayesian':
            if self.bayes_feat is None:
                self.bayes_feat = self.__get_bayes_feat(self.history["grid_idx"])
            if self.bayes_feat.shape[0] != len(self.history["grid_idx"]):
                self.bayes_feat = self.__get_bayes_feat(self.history["grid_idx"])
            # Define gaussian model
            rbf_kernel = RBF(length_scale=self.rbf_length_scale)
            self.bayes_model = GaussianProcessRegressor(kernel=rbf_kernel, normalize_y=True)
            # Fit model
            self.bayes_model.fit(self.bayes_feat, self.history["score"])
            
        print("\nOPTIMIZATION (" + approach + ")\n")
        
        # Start optimization
        for i in range(max_iter):
            print("===========================================================")
            print("ITERATION " + str(i+1) + "/" + str(max_iter) + "...\n")
            
            # Get params
            idx, hyparams = self.__pick_params(approach)
            
            # Evaluate function
            score, info = self.func(hyparams=hyparams, args=self.args)
            if (info is not None) and (type(info)!=dict):
                raise Exception("History must be None or type dict")
                
            # Update history
            self.history["grid_idx"].append(idx)
            self.history["score"].append(score)
            self.history["info"].append(info)

            # Update best model
            if self.obj == 'maximize':
                best_score = np.max(self.history["score"])
                best_idx = self.history["grid_idx"][np.argmax(
                    self.history["score"]
                )]
            else:
                best_score = np.min(self.history["score"])
                best_idx = self.history["grid_idx"][np.argmin(
                    self.history["score"]
                )]

            # Print info
            print()
            if approach == 'bayesian':
                pred = self.bayes_model.predict(self.__get_bayes_feat([idx]))[0]
                print("Bayes estim\t= %.5f" % (pred) + ", grid_idx = " +
                      str(idx))
            print("Score \t\t= %.5f" % (score) + ", grid_idx = " +
                  str(idx))
            print("Best score \t= %.5f" % (best_score) + ", grid_idx = " +
                  str(best_idx))
            print("===========================================================")
            print()
                
            # Update bayesian model
            if approach == 'bayesian':
                self.bayes_feat = np.concatenate((self.bayes_feat,
                                                  self.__get_bayes_feat([idx])),
                                                 axis=0)
                self.bayes_model.fit(self.bayes_feat, self.history["score"])
                
            # Save history
            if save_history:
                with open(self.optim_dir + 'history.pkl', 'wb') as file:
                    pickle.dump(self.history, file)
                # Save json history (human-readable)
                with open(self.optim_dir + 'history.json', 'w') as file:
                    file.write(json.dumps(self.history, indent=4))
        
        # Return
        return best_idx, best_score, self.grinder.get_hyparams(best_idx)
        
    def get_best_hyparams(self):
        # Get best model
        if len(self.history["score"]) > 0:
            if self.obj == 'maximize':
                best_score = np.max(self.history["score"])
                best_idx = self.history["grid_idx"][np.argmax(
                    self.history["score"]
                )]
            else:
                best_score = np.min(self.history["score"])
                best_idx = self.history["grid_idx"][np.argmin(
                    self.history["score"]
                )]
            
            return best_score, best_idx, self.grinder.get_hyparams(best_idx)
        else:
            raise Exception("The optimization has not started")
        


