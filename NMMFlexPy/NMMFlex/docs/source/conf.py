# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'NMMFlex'
copyright = '2023, Crhistian Cardona'
author = 'Crhistian Cardona'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon'
              ]

templates_path = ['_templates']
exclude_patterns = []


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Display todos by setting to True
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# 'alabaster', "classic", 'traditional', 'nature', 'sphinxdoc', 'bizstyle', 'pyramid'
html_theme = "nature"

html_theme_options = {
    "sidebarwidth": "30%"
}

html_logo = 'NMMFlexPy.png'
html_show_sourcelink = False
html_static_path = ['_static']
