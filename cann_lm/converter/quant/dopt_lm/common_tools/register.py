#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script description:register_manager Script.
# Copyright Huawei Technologies Co., Ltd. 2010-2027. All rights reserved.

import os
import sys
import importlib
from dopt_lm.common_tools.log import Logger as logger


class Register:
    """ Register """

    def __init__(self, name=None):
        self.name = name
        self.__feature_func_dict = {}

    def __getitem__(self, item):
        return self.__feature_func_dict.get(item)

    def __contains__(self, item):
        return item in self.__feature_func_dict.keys()

    @staticmethod
    def file_import(
        file_path,
        recursive=False,
        skip_libs_set=tuple()
    ):
        """ register file import """
        pwd = os.path.realpath(file_path)

        def get_pkg_path(path):
            name = os.path.realpath(path)
            for sys_path in sys.path:
                sys_path = os.path.realpath(sys_path)
                if path.startswith(sys_path):
                    name = path[len(sys_path) + 1:]
                    break

            name_list = name.split(os.path.sep)
            while (name_list and name_list[0] == ''):
                name_list = name_list[1:]
            pkg_name = '.'.join(name_list)
            return pkg_name

        def import_py_so_file(py_file):
            pkg_name = get_pkg_path(
                os.path.dirname(py_file)
            )
            if pkg_name:
                name_module = '.'.join(
                    [
                        pkg_name,
                        os.path.basename(py_file)[:-3]
                    ]
                )
            else:
                name_module = os.path.basename(py_file)[:-3]

            try:
                importlib.import_module(name_module)
            except Exception as ex:
                logger.warning(ex)
            finally:
                skip_warning = True

        if os.path.isfile(pwd):
            import_py_so_file(pwd)
            return

        for root, _, py_files in os.walk(pwd, topdown=True):
            if root.endswith('__pycache__'):
                continue

            for py_file in py_files:
                if not (
                    py_file.endswith('.py')
                    or
                    py_file.endswith('.so')
                ):
                    continue
                if py_file in skip_libs_set:
                    continue
                if (not py_file.endswith('__.py')) and (py_file != __file__):
                    py_file = os.path.join(root, py_file)
                    import_py_so_file(py_file)

            if not recursive:
                break

    def register_func(self, strategy_name=None):
        """ Register function"""
        def decorator(func):
            if not callable(func):
                raise Exception("registerd {} should be callable".format(func))

            name = strategy_name if strategy_name else func.__name__
            self.__feature_func_dict[name] = func
            return func
        return decorator

    def get(self, key):
        if key not in self.__feature_func_dict.keys():
            raise Exception(
                "not supported {}".format(key)
            )
        return self.__feature_func_dict.get(key)

    def get_keys(self):
        return self.__feature_func_dict.keys()