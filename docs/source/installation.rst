.. _installation:

.. Modify also the the README.md if you change docs/installation.rst

============
Installation
============

Latest release
--------------

This library is distributed on PyPI_ and
can be installed with ``pip``.

The latest release is version ``0.0.3``.

.. code:: console

   $ pip install tft-torch

The command above will automatically install all the dependencies listed in ``requirements.txt``.

.. _PyPI:  https://pypi.org/project/tft-torch

Info for developers
-------------------

The source code of the project is available on GitHub_.

.. code:: console

   $ git clone https://github.com/PlaytikaResearch/tft-torch.git

You can install the library and the dependencies from the source code with one of the following commands:

.. code:: console

   $ pip install .                        # install library + dependencies
   $ pip install .[develop]               # install library + dependencies + developer-dependencies
   $ pip install -r requirements.txt      # install dependencies
   $ pip install -r requirements-dev.txt  # install dependencies + developer-dependencies

.. _GitHub: https://github.com/PlaytikaResearch/tft-torch

For creating the "*pip-installable*" ``*.whl`` file, run the following command (at the root of the
repository):

.. code:: console

   $ python -m build

For creating the HTML documentation of the project, run the following commands:

.. code:: console

   $ cd docs
   $ make clean
   $ make html

Run tests
---------

Tests can be executed with ``pytest`` running the following commands:

.. code:: console

   $ cd tests
   $ pytest                                      # run all tests
   $ pytest test_testmodule.py                   # run all tests within a module
   $ pytest test_testmodule.py -k test_testname  # run only 1 test
