import itertools
from typing import Any, List
import copy
import matplotlib.pyplot as plt
import os
import numpy as np

from .benchmark_utils import benchmark_forward, benchmark_backward, \
                            benchmark_memory, benchmark_fwd_bwd, efficiency

class Benchmark:
    def __init__(self):
        self.parameters = {}
        self.func = {}

    @classmethod
    def parametrize(cls, param_name: str, values: List[Any]):
        def decorator(func):
            if not hasattr(func, '_benchmark'):
                func._benchmark = cls()
            func._benchmark.parameters[param_name] = values
            func._benchmark.func = func
            return func
        return decorator

    def _plot_graphics(self, results, path_graphics, key_split, memory=False, is_flops=False):

        dirname = path_graphics if path_graphics is not None else 'benchmark_results'
        os.makedirs(dirname, exist_ok=True)

        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        index_split = param_names.index(key_split)
        split_values = param_values[index_split]

        # extract function names
        names = list(list(results.values())[0].keys())

        # remove split key
        param_values.pop(index_split)
        param_names_without_split = [name for name in param_names if name != key_split]

        modes = ['fwd', 'bwd']
        if memory:
            modes.append('memory')

        for mode in modes:
            # compute the max value to set the ylim with a bit of margin
            max_value = max([x[mode] for xs in results.values() for x in xs.values()])
            max_value += 20/100 * max_value

            for combination in itertools.product(*param_values):
                combination = list(combination)

                values_per_func = {}
                for name in names:
                    values_per_func[name] = []

                for split in split_values:
                    comb = copy.copy(combination)
                    comb.insert(index_split, split)

                    param_key = str(dict(zip(param_names, comb)))
                    result = results[param_key]

                    for name in names:
                        values_per_func[name].append(round(result[name][mode], 1))

                x = np.arange(len(split_values))  # the label locations
                width = 1.0 / (len(names) + len(split_values) - 1)  # the width of the bars
                multiplier = 0

                fig, ax = plt.subplots(layout='constrained')

                for attribute, measurement in values_per_func.items():
                    offset = width * multiplier
                    rects = ax.bar(x + offset, measurement, width, label=attribute)
                    ax.bar_label(rects, padding=3)
                    multiplier += 1

                # Add some text for labels, title and custom x-axis tick labels, etc.
                string_params = ' '.join([str(x) + " " + str(combination[i]) for i, x in enumerate(param_names_without_split)])
                filename_params_str = ','.join([str(x) + "-" + str(combination[i]) for i, x in enumerate(param_names_without_split)])
                plt.xlabel('Sequence Length')
                if mode == 'memory':
                    plt.ylabel('Memory (MB)')
                elif is_flops:
                    plt.ylabel('TFLOPS')
                else:
                    plt.ylabel('Time (ms)')
                plt.title(f'{mode.upper()}\n' + string_params, wrap=True)
                ax.set_xticks(x + width, split_values)
                ax.legend(loc='upper left', ncols=2)
                ax.set_ylim(0, max_value)


                filename = dirname + "/" + f'{mode.upper()}' + "-" + filename_params_str + ".png"
                plt.savefig(filename)
                plt.close()

    def run(self, flops=None, mode="fwd", memory=False, export_csv=True, export_graphics=False, key_split=None, path_graphics=None):
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        results = {}

        for combination in itertools.product(*param_values):
            # get functions list
            params = dict(zip(param_names, combination))
            funcs = self.func(**params)
            for func_name, func in funcs.items():
                if str(params) not in results:
                    results[str(params)] = {}
                results[str(params)][func_name] = {}
                if mode == "fwd":
                    result = benchmark_forward(func, verbose=False)
                    if flops is not None:
                        flops_op = flops(**params, mode=mode)
                        results[str(params)][func_name]["fwd"] = efficiency(flops_op, result[1].mean)
                    else:
                        results[str(params)][func_name]["fwd"] = result[1].mean * 1000
                    result = benchmark_backward(func, backward=True, verbose=False)
                    if flops is not None:
                        flops_op = flops(**params, mode=mode)
                        results[str(params)][func_name]["bwd"] = efficiency(flops_op, result[1].mean)
                    else:
                        results[str(params)][func_name]["bwd"] = result[1].mean * 1000
                elif mode == "fwd_bwd":
                    result = benchmark_fwd_bwd(func, verbose=False)
                    if flops is not None:
                        flops_fwd = flops(**params, mode="fwd")
                        flops_bwd = flops(**params, mode="bwd")
                        results[str(params)][func_name]["fwd"] = efficiency(flops_fwd, result[0][1].mean)
                        results[str(params)][func_name]["bwd"] = efficiency(flops_bwd, result[1][1].mean)
                    else:
                        results[str(params)][func_name]["fwd"] = result[0][1].mean * 1000
                        results[str(params)][func_name]["bwd"] = result[1][1].mean * 1000
                else:
                    raise ValueError(f"Invalid mode: {mode}")

                if memory:
                    result = benchmark_memory(func, verbose=False)
                    results[str(params)][func_name]["memory"] = result * 1024

        if export_graphics:
            self._plot_graphics(results, path_graphics, key_split=key_split, memory=memory, is_flops=flops is not None)

        return results
