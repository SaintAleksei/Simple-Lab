#!/usr/bin/python3

# TODO Add pretty printing of errors (including Schema)
# TODO Desctiption/Comments

import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from schema import Schema, Optional, And, Or
from csv import reader as csv_reader
from yaml import full_load as yaml_load
from sys import argv as sys_argv
from collections import deque
from json import dumps as json_dumps

class SimplePlot:
  '''Simple matplotlib.pyplot wrapper'''

  def __init__(self, params=(), kwparams={}):
    '''Initialize instance

    Arguments:
      params - pyplot.subplots params
      kwparams - pyplot.subplots kwargs'''

    fig, ax = plt.subplots(*params, **kwparams)
    self.ax = ax
    self.fig = fig
    self.default_markerstyles = deque(['o', 'v', '^', '<', '>'])
    self.default_linestyles   = deque(['solid', 'dotted', 'dashed', 'dashdot'])

  def set_grid(self, xaxis, yaxis):
    '''Grid initialization

    Arguments:
      xaxis - X axis sampling
      yaxis - Y axis sampling

    Sampling is accepted in format (start, end, major_ticks, minor_ticks)'''

    self.ax.set_xticks(np.linspace(xaxis[0], xaxis[1], xaxis[2] + 1))
    self.ax.set_xticks(np.linspace(xaxis[0], xaxis[1],\
                       xaxis[2] * xaxis[3] + 1), minor=True)
    self.ax.set_yticks(np.linspace(yaxis[0], yaxis[1], yaxis[2] + 1))
    self.ax.set_yticks(np.linspace(yaxis[0], yaxis[1],\
                       yaxis[2] * yaxis[3] + 1), minor=True)
    self.ax.grid(visible=True, which='major', color='k',\
                 linestyle='-', linewidth=1)
    self.ax.grid(visible=True, which='minor', color='k',\
                 linestyle='--', linewidth=0.5)

  def plot_data(self, xdata, ydata, xerr=None, yerr=None, kwparams=None):
    '''Plot data with errors

    Arguments:
      xdata    - X axis data
      ydata    - Y axis data
      xerr     - X axis data errors
      yerr     - Y axis data errors
      kwparams - matplotlib.axes.Axes.errorbar kwargs'''

    try:
      if kwparams is None:
        kwparams = self.data_default()
      to_return = self.ax.errorbar(xdata, ydata, xerr, yerr, **kwparams);
      self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),\
                     ncol= 2, fontsize='large')
      return to_return
    except:
      print('Can\'t plot data')
      raise
    
  def plot_func(self, func, xrang, kwparams=None):
    '''Plot function

    Arguments:
      func      - callable to plot
      xrang     - tuple with plot range (x_start, x_end)
      kwaparams - matplotlib.axes.Axes.plot kwargs'''

    try:
      if kwparams is None:
        kwparams = self.func_default()

      x = np.linspace(xrang[0], xrang[1], 1000)
      y = func(x)
      to_return = self.ax.plot(x, y, markersize=0, **kwparams)
      self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),\
                     ncol= 2, fontsize='large')
      return to_return
    except:
      print('Can\'t plot func')
      raise

  def data_default(self):
    '''Default kwparams for plot_data'''

    markerstyle = self.default_markerstyles.pop()
    self.default_markerstyles.appendleft(markerstyle)
    return {
      'markersize': 2,
      'marker': markerstyle,
      'color': 'k',
      'linewidth': 0,
      'elinewidth': 0.5,
    }
  
  def func_default(self):
    '''Default kwparams for plot_func'''

    linestyle = self.default_linestyles.pop()
    self.default_linestyles.appendleft(linestyle)
    return {
      'linewidth': 1,
      'linestyle': linestyle,
      'color': 'k'
    }

  @staticmethod
  def draw():
    '''Almost useless wrapper :)'''

    plt.show()

class Namespace:
  ''' 'Namespace' of 'Values'
  
    It is reprsented in pairs of 'Name' : 'Value',
  in fact, it is just little bit clever default 
  python dictionary'''

  def __init__(self):
    self.__dict = {}

  def __getitem__(self, key):
    return self.__dict[key]

  def __setitem__(self, key, value):
    '''Wrapper with type checking'''
    if type(value) is not Value:
      raise TypeError('Object of type \'Value\' required')
    self.__dict[key] = value

  def keys(self):
    return self.__dict.keys()

  def copy(self):
    new = Namespace()
    new.__dict = self.__dict.copy()
    return new

  def to_json(self):
    '''JSON serializer

    Return text representation of 'Namespace' '''

    scalars = {}
    for k, v in self.__dict.items():
      if not v.is_vector():
        scalars[k] = v.to_strings()[0]
    return json_dumps(scalars, indent=4, sort_keys=True)

class Value:
  '''Representation of 'Value'
  
  'Value' consist of val (value itself) and error (error of value)'''

  def __init__(self, val, err=0.0):
    '''Initialize instance

    Arguments:
      val - value itself (scalar or vector)
      err - error of value (scalar or vector)
    '''

    try:
      iter(val)
    except TypeError:
      val = [val]

    try:
      iter(err)
    except TypeError:
      err = [err]

    try:
      self.__val = np.array(val, dtype=np.float64)
      if len(val) != 1 and len(err) == 1:
        self.__err = np.full(len(val), iter(err).__next__(), dtype=np.float64)
      else:
        self.__err = np.array(err, dtype=np.float64)

      if self.__val.ndim != 1 or\
         self.__err.ndim != 1 or\
         self.__val.size != self.__err.size:
        raise ValueError('Bad arguments')

      self.__val.setflags(write=False)
      self.__err.setflags(write=False)
    except:
      print("Can't create Value")
      raise

  def __str__(self):
    return '\n'.join(self.to_strings)

  def is_vector(self):
    '''Check if 'Value' is vector'''
    return not self.__val.size == 1

  def to_strings(self):
    '''Serialize 'Value' to strings

    Return list of strings representing each value from 'Value' vector'''

    return [f'{val:.7e}+-{err:.7e}' for val, err in zip(self.__val, self.__err)]

  @property
  def value(self):
    return self.__val

  @property
  def error(self):
    return self.__err

  @staticmethod
  def from_strings(strings):
    '''Unserialize 'Value' from strings

    Arguments:
      strings - strings that shoud be unserialized

    Return new unserialized 'Value' '''

    val = []
    err = []
    for s in strings:
      if type(s) is not str:
        TypeError('Object of type \'str\' required')
      splited = s.split('+-')
      try:
        if len(splited) != 2:
          ValueError('Bad argument')
        v, e = float(splited[0]), float(splited[1])
        val.append(v)
        err.append(e)
      except ValueError:
        print(f'Can\'t parse string \'{s}\'')
        raise
    return Value(val, err)

  @staticmethod
  def merge(left, right):
    '''Merge two values

    Arguments:
      left - first 'Value' to merge
      right - second 'Value' to merge

    Return new merged 'Value' '''

    if type(left) is not Value or type(right) is not Value:
      raise TypeError('Objects of type \'Value\' required')

    new_val = np.concatenate((left.val, right.val))
    new_err = np.concatenate((left.error, right.error))
    return Value(new_val, new_err)

  @staticmethod
  def slice(*args, rang):
    '''Slice list of values

    Arguments:
      args - list of 'Values' to slice
      rang - range values for slicing in format (start, end)

    Return list of new sliced 'Values' '''

    new_vals = [[] for _ in range(len(args))]
    new_errs = [[] for _ in range(len(args))]
    try:
      for i, v in enumerate(args[0].value):
        if rang[0] <= v <= rang[1]:
          for j, a in enumerate(args):
            new_vals[j].append(a.value[i])
            new_errs[j].append(a.error[i])
      result = []
      for v, e in zip(new_vals, new_errs):
        result.append(Value(v, e))
      return result
    except:
      print('Can\'t slice vals')
      raise

class SimpleLab:
  '''Main API Class
  
  There a two independent ways of using it:
     - Complete program. Just one call to SimpleLab.process_all
      performs complete project. Ususaly with data from YAML file
     - Python API. Dierct SimpleLab API using with appropiate input objects.

  Of course, this ways can be mixed'''


  # SimplePlot instance
  plot = None
  # Global 'Namespace'
  namespace = Namespace()

  # I used Schema for input validation in all SimpleLab API
  schem = {}
  schem['plot'] = {
    Optional('title'): str,
    Optional('xlabel'): str,
    Optional('ylabel'): str,
    Optional('grid'): And([[float, int]], lambda l: len(l) == 2 and\
                      len(l[0]) == len(l[1]) == 4)
  }
  schem['value'] = {
    'val': Or(str, float, int, [str, float, int]),
    Optional('err'): Or(str, float, int, [str, float, int])
  }
  schem['file'] = str
  schem['fit'] = {
    'func': str,
    'var': str,
    Optional('params'): [str],
    Optional('plot_range'): And([float, int], lambda l: len(l) == 2),
    Optional('plot_params'): {
      str: object
    },
    Optional('fit_range'): And([float, int], lambda l: len(l) == 2),
    Optional('fit_params'): {
      str: object
    },
    Optional('fit_guess'): {
      str: Or(float, int)
    }
  }
  schem['data'] = {
    'xdata': schem['value'],
    'ydata': schem['value'],
    Optional('plot'): {
      str: object
    },
    Optional('fit'): [schem['fit']]
  }
  schem['func'] = {
    'func': str,
    'var': str,
    'range': And([float, int], lambda l: len(l) == 2),
    Optional('plot'):{
      str: object
    }
  }
  schem['export'] = str
  schem['config'] = {
    Optional('plot'): schem['plot'],
    Optional('files'): [schem['file']],
    Optional('values'): {
      str: schem['value']
    },
    Optional('data'): [schem['data']],
    Optional('funcs'): [schem['func']],
    Optional('exports'): [schem['export']]
  }

  def __init__(self):
    '''Instance initialization'''

    self.namespace = SimpleLab.namespace.copy()
    
  def process_all(self, config):
    '''Perform complete project. Used when running as complete program

    Argumets:
      config - well-formed dictionary describing the project

    Config consist of optional fields:
      plot   - main plot parameters. See SimpleLab.process_plot for details
      files  - list of .csv file to import into local 'Namespace'. 
              See SimpleLab.process_file for details
      values - 'Name' : 'Value' pairs. See SimpleLab.process_value for details
      data   - list of 'data' units. See SimpleLab.process_data for details
      funcs  - list of functions to plot. 
              See SimpleLab.process_func for details

    Config usually comes from YAML file'''

    # config validation
    Schema(self.schem['config']).validate(config)

    # 'plot' field processing
    self.process_plot(config.get('plot', {}))

    # 'files' field processing
    for file_name in config.get('files', []):
      self.process_file(file_name)

    # 'values' field processing
    for k, v in config.get('values', {}).items():
      self.namespace[k] = self.process_value(v)

    # 'data' field processing
    for data in config.get('data', []):
      self.process_data(data)

    # 'funcs' field processing
    for func in config.get('funcs', []):
      self.process_func(func)

    # 'exports' fielts processing
    for export in config.get('exports', []):
      self.process_export(export)

  def process_plot(self, plot):
    '''Main plot parameters processing

    Arguments:
      plot - appropriate dictionary

    Fields:
      title  - main plot title
      xlabel - X axis label
      ylabel - Y axis label
      grid   - see SimplePlot.set_grid for details

    One SimpleLab instance have one plot instance'''

    Schema(self.schem['plot']).validate(plot)

    if self.plot is not None:
      return

    self.plot = SimplePlot()
    self.plot.ax.set_title(plot.get('title', ''), fontsize='xx-large')
    self.plot.ax.set_xlabel(plot.get('xlabel', ''),\
                            fontsize='x-large', loc='right')
    self.plot.ax.set_ylabel(plot.get('ylabel', ''), fontsize='x-large',\
                            loc='top', rotation='horizontal')
    grid = plot.get('grid', None)
    if grid is not None:
      self.plot.set_grid(*grid)

  def process_data(self, data):
    '''Data unit processing

    Arguments:
      data - appropriate dictionary

    Fields:
      xdata - X axis 'Value'. See SimpleLab.process_value for details
      ydata - Y axis 'Value'. See SimpleLab.process_value for details
      plot  - dictionaty with plot params. See SimplePlot.plot_data for details
      fit   - fit unit. See SimpleLab.process_fit for details'''
      
    Schema(self.schem['data']).validate(data)

    x_val = self.process_value(data['xdata'])
    y_val = self.process_value(data['ydata'])

    kwparams = data.get('plot', None)
    if kwparams is None:
      kwparams = {} 
      kwparams['label'] = 'Experimental data'
      kwparams |= self.plot.data_default()

    self.plot.plot_data(x_val.value, y_val.value,\
                        y_val.error, x_val.error,\
                        kwparams=kwparams)

    for fit in data.get('fit', []):
      self.process_fit(fit, x_val, y_val)

  def process_file(self, file_name):
    '''File processing

    Arguments: 
      file_name - name of file

    Import vectors to the local 'Namespace' with default errors
  
    Only CSV format is available for now. See Helpers.parse_csv'''

    Schema(self.schem['file']).validate(file_name)

    dict_csv = Helpers.parse_csv(file_name)
    for k, v in dict_csv.items():
      self.namespace[k] = Value(v)

  def process_value(self, value):
    '''Creating 'Value' from appropriate dictionary

    Arguments:
      value - appropriate dictionary
  
    Fields:
      val - value representation
      err - value representation. Optional

      Value representation should be float, int, string of list of them.
    In case with int or float it is value itself. In case of string it should
    be formula that sympy can understand. Anything from local 'Namespace'
    can be used in this formula. Formula calculation is performed by
    Helpers.compute_value function. If err is ommited 'Value' error will
    be default in case of float or int, and it will be computed using
    error theory in case of sympy formula. If err is presented, is will be 
    used as value representation for 'Value' error.

    Return new 'Value' '''
    
    Schema(self.schem['value']).validate(value)

    try:
      val = value['val']
      val_parsed = np.empty(0, dtype=np.float64)
      err_parsed = np.empty(0, dtype=np.float64)
      if type(val) is list:
        for v in val:
          if type(v) is str:
            new = Helpers.compute_value(v, self.namespace)
            val_parsed = np.concatenate((val_parsed, new.value))
            err_parsed = np.concatenate((err_parsed, new.error))
          elif type(v) is float or type(v) is int:
            val_parsed = np.append(val_parsed, v)
            err_parsed = np.append(err_parsed, 0.0)
          else:
            raise ValueError('Bad argument')
      elif type(val) is str:
        new = Helpers.compute_value(val, self.namespace)
        val_parsed = np.concatenate((val_parsed, new.value))
        err_parsed = np.concatenate((err_parsed, new.error))
      elif type(val) is float or type(val) is int:
        val_parsed = np.append(val_parsed, val)
        err_parsed = np.append(err_parsed, 0.0)
      else:
        raise ValueError('Bad argument')

      err = value.get('err', None)
      if err is None:
        return Value(val_parsed, err_parsed)

      err_parsed = np.empty(0, dtype=np.float64)

      if type(err) is list:
        for v in err:
          if type(v) is str:
            new = Helpers.compute_value(v, self.namespace)
            err_parsed = np.concatenate((err_parsed, new.value))
          elif type(v) is float or type(v) is int:
            err_parsed = np.append(err_parsed, v)
          else:
            raise ValueError('Bad argument')
      elif type(err) is str:
        new = Helpers.compute_value(err, self.namespace)
        err_parsed = np.concatenate((err_parsed, new.value))
      elif type(err) is float or type(err) is int:
        err_parsed = np.append(err_parsed, err)
      else:
        raise ValueError('Bad argument')

      return Value(val_parsed, err_parsed)
    except:
      print('Can\'t process value\n')
      raise

  def process_fit(self, fit, x_val, y_val):
    '''Processing fit

    Arguments:
      fit   - appropriate dictionary
      x_val - X axis 'Value'
      y_val - Y axis 'Value' 

    fit fields:
      func - string representation of fit function in sympy format 
      var  - name of independent variable in func
      params - list of 'Names' of parameters that should be imported as
              'Values' into the local 'Namespace'
      fit_guess - guess of fit parameters. Sometimes it is necessary
      fit_params - fit calculation parameters. See scipy.optimize.curve_fit
      fit_range  - independent variable range at which fit should be computed
      plot_params - fit plot parameters. See SimplePlot.plot_func
      plot_range  - independent variable range aw which curve should be drawn'''
    
    Schema(Value).validate(x_val)
    Schema(Value).validate(y_val)
    Schema(self.schem['fit']).validate(fit)

    try:
      plot_params = fit.get('plot_params', None)
      fit_range = fit.get('fit_range', (min(x_val.value), max(x_val.value)))
      plot_range = fit.get('plot_range', None)
      if plot_range is None:
        plot_range = fit_range
      x, y = Value.slice(x_val, y_val, rang=fit_range)

      fit_guess = fit.get('fit_guess', {})
      params = fit.get('params', {})
      var = fit['var']

      func_expr = sympy.sympify(fit['func'])
      free_syms = [str(sym) for sym in func_expr.free_symbols]
      free_syms.remove(var)
      func = sympy.lambdify([var] + free_syms, func_expr, 'numpy')

      fit_params = fit.get('fit_params', {})
      fit_params['p0'] = [fit_guess.get(k, 1.0) for k in free_syms]
      popt, pcov = curve_fit(func, x.value, y.value, sigma=y.error, absolute_sigma=True, **fit_params)
      errs = np.sqrt(np.diag(pcov))

      for k, val, err in zip(free_syms, popt, errs):
        if k in params:
          self.namespace[k] = Value(val, err)
          
      for k, v in zip(params, popt):
        # FIXME This is the best try for real round :)
        evalfed = float(f'{sympy.sympify(k).evalf(3, subs={k : v}):.3e}')
        func_expr = func_expr.subs(k, sympy.UnevaluatedExpr(evalfed))

      if plot_params is None:
        plot_params = {}

      plot_params['label'] = '$' + sympy.latex(func_expr, mul_symbol='dot') + '$'
      plot_params.update(self.plot.func_default())
      
      func_fit = lambda x: func(x, *popt)
      self.plot.plot_func(func_fit, plot_range, kwparams=plot_params)
    except:
      print('Can\'t fit data')
      raise

  def process_export(self, export):
    '''Export processing

    Arguments:
      export - 'Name' of 'Value' to export'''

    Schema(self.schem['export']).validate(export)

    SimpleLab.namespace[export] = self.namespace[export]

class Helpers:
  '''Helpfull functions'''

  @staticmethod
  def compute_value(expr, namespace=None):
    '''Computing value

    Arguments:
      expr - string expression that can be passed to sympy.sympify
      namespace - 'Namespace' with 'Values' which 'Names' can be 
                 used in expr string

    Return computeed 'Value' '''

    if namespace is None:
      namespace = Namespace()

    if type(namespace) is not Namespace:
      raise ValueError('Bad namespace')

    sympy_locals = {k: sympy.Symbol(k) for k in namespace.keys()}
    try:
      expr = sympy.sympify(expr, locals=sympy_locals)
    except:
      print(f'Can\'t parse expression \'{expr}\'')
      raise

    scalars = {}
    vectors = {}
    errors = {}
    syms = [str(sym) for sym in expr.free_symbols]
    for sym in expr.free_symbols:
      sym = str(sym)
      try:
        value = namespace[sym]
      except KeyError:
        print(f'Value \'{sym}\' is not found in the namespace')
        raise

      if value.is_vector():
        vectors[sym] = value.value
      else:
        scalars[sym] = value.value[0]

      errors[sym] = value.error

    try:
      args = [v for v in vectors.values()]
      expr_subs = expr.subs([(k, v) for k, v in scalars.items()])
      expr_func = sympy.lambdify(vectors.keys(), expr_subs, 'numpy')
      value = expr_func(*args)

      diff_expr = {k: sympy.diff(expr, k).evalf(subs=scalars) for k in (scalars | vectors).keys()}
      diff_funcs = {k: sympy.lambdify(vectors.keys(), diff_expr[k], 'numpy') for k in diff_expr.keys()}
      error = np.sqrt(sum([ (f(*args) * errors[k]) ** 2 for k, f in diff_funcs.items()]))
    except:
      print(f'Can\'t compute value \'{expr}\'')
      raise
    
    return Value(value, error)

  @staticmethod
  def parse_csv(fname):
    '''Simple CSV parser

    Arguments:
      fname - name of file

    Return dictionary representation of CSV file'''

    with open(fname, 'r') as file_csv:
      reader = csv_reader(file_csv, delimiter=',')
      names = reader.__next__()
      result = {k: [] for k in names}
      for row in reader:
        for i, name in enumerate(names):
          result[name].append(float(row[i]))

    return result

if __name__ == '__main__':
  for arg in sys_argv[1:]:
    with open(arg, 'r') as f:
      yaml_data = yaml_load(f)
    lab = SimpleLab()
    lab.process_all(yaml_data)
  print(SimpleLab.namespace.to_json())
  SimplePlot.draw()
