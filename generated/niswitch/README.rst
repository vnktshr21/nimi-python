Overall Status
--------------

+----------------------+------------------------------------------------------------------------------------------------------------------------------------+
| master branch status | |BuildStatus| |MITLicense| |CoverageStatus|                                                                                        |
+----------------------+------------------------------------------------------------------------------------------------------------------------------------+
| GitHub status        | |OpenIssues| |OpenPullRequests|                                                                                                    |
+----------------------+------------------------------------------------------------------------------------------------------------------------------------+

===========  ============================================================================================================================
Info         NI Modular Instrument driver APIs for Python.
Author       NI
===========  ============================================================================================================================

.. |BuildStatus| image:: https://api.travis-ci.com/ni/nimi-python.svg
    :alt: Build Status - master branch
    :target: https://travis-ci.org/ni/nimi-python

.. |MITLicense| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :alt: MIT License
    :target: https://opensource.org/licenses/MIT

.. |CoverageStatus| image:: https://codecov.io/github/ni/nimi-python/graph/badge.svg
    :alt: Test Coverage - master branch
    :target: https://codecov.io/github/ni/nimi-python

.. |OpenIssues| image:: https://img.shields.io/github/issues/ni/nimi-python.svg
    :alt: Open Issues + Pull Requests
    :target: https://github.com/ni/nimi-python/issues

.. |OpenPullRequests| image:: https://img.shields.io/github/issues-pr/ni/nimi-python.svg
    :alt: Open Pull Requests
    :target: https://github.com/ni/nimi-python/pulls


.. _about-section:

About
=====

The **niswitch** module provides a Python API for NI-SWITCH. The code is maintained in the Open Source repository for `nimi-python <https://github.com/ni/nimi-python>`_.

Support Policy
--------------
**niswitch** supports all the Operating Systems supported by NI-SWITCH.

It follows `Python Software Foundation <https://devguide.python.org/#status-of-python-branches>`_ support policy for different versions of CPython.

NI created and supports **niswitch**.


NI-SWITCH Python API Status
---------------------------

+-------------------------------+-------------------------+
| NI-SWITCH (niswitch)          |                         |
+===============================+=========================+
| Driver Version Tested Against | 2025 Q1                 |
+-------------------------------+-------------------------+
| PyPI Version                  | |niswitchLatestVersion| |
+-------------------------------+-------------------------+
| Supported Python Version      | |niswitchPythonVersion| |
+-------------------------------+-------------------------+
| Documentation                 | |niswitchDocs|          |
+-------------------------------+-------------------------+
| Open Issues                   | |niswitchOpenIssues|    |
+-------------------------------+-------------------------+
| Open Pull Requests            | |niswitchOpenPRs|       |
+-------------------------------+-------------------------+


.. |niswitchLatestVersion| image:: http://img.shields.io/pypi/v/niswitch.svg
    :alt: Latest NI-SWITCH Version
    :target: http://pypi.python.org/pypi/niswitch


.. |niswitchPythonVersion| image:: http://img.shields.io/pypi/pyversions/niswitch.svg
    :alt: NI-SWITCH supported Python versions
    :target: http://pypi.python.org/pypi/niswitch


.. |niswitchDocs| image:: https://readthedocs.org/projects/niswitch/badge/?version=latest
    :alt: NI-SWITCH Python API Documentation Status
    :target: https://niswitch.readthedocs.io/en/latest


.. |niswitchOpenIssues| image:: https://img.shields.io/github/issues/ni/nimi-python/niswitch.svg
    :alt: Open Issues + Pull Requests for NI-SWITCH
    :target: https://github.com/ni/nimi-python/issues?q=is%3Aopen+is%3Aissue+label%3Aniswitch


.. |niswitchOpenPRs| image:: https://img.shields.io/github/issues-pr/ni/nimi-python/niswitch.svg
    :alt: Pull Requests for NI-SWITCH
    :target: https://github.com/ni/nimi-python/pulls?q=is%3Aopen+is%3Aissue+label%3Aniswitch



.. _niswitch_installation-section:

Installation
------------

As a prerequisite to using the **niswitch** module, you must install the NI-SWITCH runtime on your system. Visit `ni.com/downloads <http://www.ni.com/downloads/>`_ to download the driver runtime for your devices.

The nimi-python modules (i.e. for **NI-SWITCH**) can be installed with `pip <http://pypi.python.org/pypi/pip>`_::

  $ python -m pip install niswitch


Contributing
============

We welcome contributions! You can clone the project repository, build it, and install it by `following these instructions <https://github.com/ni/nimi-python/blob/master/CONTRIBUTING.md>`_.

Usage
------

The following is a basic example of using the **niswitch** module to open a session to a Switch and connect channels.

.. code-block:: python

    import niswitch
    with niswitch.Session("Dev1") as session:
        session.connect(channel1='r0', channel2='c0')

`Other usage examples can be found on GitHub. <https://github.com/ni/nimi-python/tree/master/src/niswitch/examples>`_

.. _support-section:

Support / Feedback
==================

For support specific to the Python API, follow the processs in `Bugs / Feature Requests`_.
For support with hardware, the driver runtime or any other questions not specific to the Python API, please visit `NI Community Forums <https://forums.ni.com/>`_.

.. _bugs-section:

Bugs / Feature Requests
=======================

To report a bug or submit a feature request specific to Python API, please use the
`GitHub issues page <https://github.com/ni/nimi-python/issues>`_.

Fill in the issue template as completely as possible and we will respond as soon
as we can.


.. _documentation-section:

Documentation
=============

Documentation is available `here <http://niswitch.readthedocs.io>`_.


.. _license-section:

License
=======

**nimi-python** is licensed under an MIT-style license (`see
LICENSE <https://github.com/ni/nimi-python/blob/master/LICENSE>`_).
Other incorporated projects may be licensed under different licenses. All
licenses allow for non-commercial and commercial use.


**gRPC Features**

For driver APIs that support it, passing a GrpcSessionOptions instance as a parameter to Session.__init__() is
subject to the NI General Purpose EULA (`see NILICENSE <https://github.com/ni/nimi-python/blob/master/NILICENSE>`_).