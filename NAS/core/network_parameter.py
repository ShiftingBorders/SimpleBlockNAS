import copy
import numpy as np
import torch


class NetParameter:

    def __init__(self, max_val_init, upper_value, ga_support, change_min_value=1):
        """
        Initialize NetParameter with initial values and settings.
        Args:
            max_val_init (int/float): Initial value for parameter.
            upper_value (int/float): Upper bound for parameter.
            ga_support (bool): Whether GA operations are supported.
            change_min_value (int/float): Minimum allowed value.
        """
        self.change_min_value = change_min_value
        if max_val_init > 0 and max_val_init < 1:
            self.val = float(np.random.rand())
            if self.val > max_val_init:
                self.val = max_val_init
        else:
            self.val = np.random.randint(1, max_val_init + 1)
        self.ga_support = ga_support
        self.upper_value = upper_value

        self.max_val_init = max_val_init

        self.LOCK = False

    def force_value(self, val, lock=True, ga_support=False):
        """
        Force the parameter to a specific value and optionally lock it.
        Args:
            val: Value to set.
            lock (bool): Whether to lock the parameter.
            ga_support (bool): Whether GA operations are supported.
        """
        self.val = val
        self.upper_value = val
        self.LOCK = lock
        self.ga_support = ga_support

    def re_randomize(self):
        """
        Randomize the parameter value if not locked.
        """
        if self.LOCK is True:
            return

        max_val_init = self.max_val_init
        if max_val_init > 0 and max_val_init < 1:
            self.val = float(np.random.rand())
            if self.val > max_val_init:
                self.val = max_val_init
        else:
            self.val = np.random.randint(1, max_val_init + 1)

    def lock(self):
        """
        Lock the parameter to prevent changes.
        """
        self.LOCK = True

    def unlock(self):
        """
        Unlock the parameter to allow changes.
        """
        self.LOCK = False

    def check_lock(self):
        """
        Check if the parameter is locked.
        Returns:
            bool: True if locked, False otherwise.
        """
        return self.LOCK

    def get_value(self):
        """
        Get the current value of the parameter.
        Returns:
            int/float: Parameter value.
        """
        return self.val

    def change_value(self, new_value, new_upper=False, avoid_self_check = False):
        """
        Change the value of the parameter, optionally updating the upper bound.
        Args:
            new_value: New value for the parameter.
            new_upper (bool): If True, update upper bound.
            avoid_self_check (bool): If True, avoid self-check.
        """
        if not isinstance(new_value,int) and not isinstance(new_value,float):
            raise ValueError(f"Tried to assigned value: {new_value} to parameter")
        if self.LOCK is False:
            if new_value < self.change_min_value:
                new_value = self.change_min_value
            if new_upper is True:
                self.upper_value = new_value
            else:
                self.val = new_value
        if not avoid_self_check:
            self.self_check()

    def self_check(self):
        """
        Check and correct the parameter value to stay within bounds.
        """
        if self.val > self.upper_value:
            self.val = self.upper_value
        if self.val < self.change_min_value:
            self.val = self.change_min_value

    def check_ga_support(self):
        """
        Check if GA operations are supported for this parameter.
        Returns:
            bool: True if supported, False otherwise.
        """
        return self.ga_support

    def get_upper_value(self):
        """
        Get the upper bound value for the parameter.
        Returns:
            int/float: Upper bound value.
        """
        return self.upper_value




class MultivaluedParameter:

    def __init__(self, values):
        """
        Initialize MultivaluedParameter with a list of values.
        Args:
            values (list): List of parameter values.
        """
        self.values = copy.deepcopy(values)

    def __str__(self):
        """
        String representation of the parameter values.
        Returns:
            str: String of values.
        """
        return f'Values: {self.values}'

    def __repr__(self):
        """
        Representation of the parameter values.
        Returns:
            str: String of values.
        """
        return f'Values: {self.values}'

    def get_value(self, id_):
        """
        Get the value at a specific index.
        Args:
            id_ (int): Index of the value.
        Returns:
            value: Value at the index or 0 if out of bounds.
        """
        if id_ > len(self.values) - 1:
            return 0
        return self.values[id_]

    def change_value(self, id_, new_val):
        """
        Change the value at a specific index.
        Args:
            id_ (int): Index of the value.
            new_val: New value to set.
        """
        self.values[id_] = copy.deepcopy(new_val)

    def get_num_parameters(self):
        """
        Get the number of parameters.
        Returns:
            int: Number of parameter values.
        """
        return len(self.values)
