#!/bin/sh

python -m PyInstaller --noconfirm \
--specpath='Tacotron' \
--add-data="models/*:models" \
--add-data="templates/*:templates" \
--hidden-import="pkg_resources.py2_warn" \
--hidden-import="sklearn.utils._cython_blas" \
--hidden-import="sklearn.neighbors.typedefs" \
--hidden-import="sklearn.neighbors.quad_tree" \
--hidden-import="sklearn.tree" \
--hidden-import="sklearn.tree._utils" \
--hidden-import="scipy.special.cython_special" \
--exclude-module="torch.distributions" \
Tacotron/Tacotron.py
