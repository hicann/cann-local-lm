#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script description:tf_common_utils Script.
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved

import os
import logging

logging.basicConfig(format="%(asctime)s [%(levelname)-6s] - %(message)s")


class Logger(object):
    handlers = dict()

    @staticmethod
    def add_file_handler(name, file_path):
        logger = logging.getLogger(__name__)

        handlers = Logger.handlers
        if name in handlers:
            Logger.logger.warning(f"{name} is registered yet")
            return
        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        handlers[name] = fh

    @staticmethod
    def remove_file_handler(name):
        logger = logging.getLogger(__name__)

        handlers = Logger.handlers
        if name not in handlers:
            Logger.logger.warning(f"{name} is not registered")
            return
        logger.removeHandler(handlers.pop(name))

    @staticmethod
    def logger(level, *msg):
        log = logging.getLogger(__name__)
        log.setLevel(logging.INFO)

        if level == "info":
            log.info(*msg)
        elif level == "debug":
            log.debug(*msg)
        elif level == "warning":
            log.warning(*msg)
        elif level == "error":
            log.error(*msg)
        else:
            log.critical(*msg)

    # @wrap_cando
    @staticmethod
    def info(*msg):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return
        Logger.logger("info", *msg)

    # @wrap_cando
    @staticmethod
    def warning(*msg):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return
        Logger.logger("warning", *msg)

    # @wrap_cando
    @staticmethod
    def debug(*msg):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return
        Logger.logger("debug", *msg)

    # @wrap_cando
    @staticmethod
    def error(*msg):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return
        Logger.logger("error", *msg)


class FileTxtLogger:
    loggers = dict()
    logger_files = dict()
    output_path = ""

    @classmethod
    def add_recoder(cls, name, file_path=None):
        if name in cls.loggers:
            return
        if file_path is None:
            file_path = name
        record_file = file_path + ".txt"
        if not file_path[0] == os.path.sep:
            record_file = os.path.join(cls.output_path, file_path + ".txt")
        os.makedirs(os.path.dirname(record_file), exist_ok=True)
        cls.loggers[name] = record_file

    @classmethod
    def record(cls, name, msg, force=False):
        if name not in cls.loggers:
            if not force:
                return
            cls.add_recoder(name)
        file_name = cls.loggers[name]
        with open(file_name, "a") as f:
            f.writelines(msg + "\n")


def set_record_mode(model, output_path, name_list=None):
    FileTxtLogger.output_path = output_path
    for name, m in model.named_modules():
        if hasattr(m, "layer_name") and m.layer_name is not None:
            FileTxtLogger.add_recoder(m.layer_name, m.layer_name)
            FileTxtLogger.add_recoder(f"{m.layer_name}_out", f"{m.layer_name}_out")


def visualize_dist(ckpt):
    for k in ckpt:
        if k.endswith("input_quantizer.s"):
            bit_name = f"{k[:-2]}.bit"
            offset_name = f"{k[:-2]}.offset"
            dist_str = f"{k:60}{ckpt[bit_name].cpu().item()}\t{ckpt[k].cpu().item():.6f}\t{ckpt[offset_name].cpu().item():.6f}\t{1 / ckpt[k].cpu().item():.6f}"
            Logger.info(dist_str)
            FileTxtLogger.record("dist", dist_str, force=True)
            continue
        if k.endswith("quantizer.s"):
            bit_name = f"{k[:-2]}.bit"
            offset_name = f"{k[:-2]}.offset"
            dist_str = f"{k:60}{ckpt[bit_name].cpu().item()}\t{ckpt[k].cpu().item():.6f}\t{ckpt[offset_name].cpu().item():.6f}\t{1 / ckpt[k].cpu().item():.6f}"
            Logger.info(dist_str)
            FileTxtLogger.record("dist_others", dist_str, force=True)
