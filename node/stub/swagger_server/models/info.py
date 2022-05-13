# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class Info(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, provider: object=None, version: object=None, arch: str=None):  # noqa: E501
        """Info - a model defined in Swagger

        :param provider: The provider of this Info.  # noqa: E501
        :type provider: object
        :param version: The version of this Info.  # noqa: E501
        :type version: object
        :param arch: The arch of this Info.  # noqa: E501
        :type arch: str
        """
        self.swagger_types = {
            'provider': object,
            'version': object,
            'arch': str
        }

        self.attribute_map = {
            'provider': 'provider',
            'version': 'version',
            'arch': 'arch'
        }

        self._provider = provider
        self._version = version
        self._arch = arch

    @classmethod
    def from_dict(cls, dikt) -> 'Info':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Info of this Info.  # noqa: E501
        :rtype: Info
        """
        return util.deserialize_model(dikt, cls)

    @property
    def provider(self) -> object:
        """Gets the provider of this Info.

        The OpenFaaS Provider  # noqa: E501

        :return: The provider of this Info.
        :rtype: object
        """
        return self._provider

    @provider.setter
    def provider(self, provider: object):
        """Sets the provider of this Info.

        The OpenFaaS Provider  # noqa: E501

        :param provider: The provider of this Info.
        :type provider: object
        """
        if provider is None:
            raise ValueError("Invalid value for `provider`, must not be `None`")  # noqa: E501

        self._provider = provider

    @property
    def version(self) -> object:
        """Gets the version of this Info.

        Version of the Gateway  # noqa: E501

        :return: The version of this Info.
        :rtype: object
        """
        return self._version

    @version.setter
    def version(self, version: object):
        """Sets the version of this Info.

        Version of the Gateway  # noqa: E501

        :param version: The version of this Info.
        :type version: object
        """
        if version is None:
            raise ValueError("Invalid value for `version`, must not be `None`")  # noqa: E501

        self._version = version

    @property
    def arch(self) -> str:
        """Gets the arch of this Info.

        Platform architecture  # noqa: E501

        :return: The arch of this Info.
        :rtype: str
        """
        return self._arch

    @arch.setter
    def arch(self, arch: str):
        """Sets the arch of this Info.

        Platform architecture  # noqa: E501

        :param arch: The arch of this Info.
        :type arch: str
        """

        self._arch = arch
