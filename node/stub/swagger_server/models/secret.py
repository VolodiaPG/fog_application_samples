# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class Secret(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, name: str=None, value: str=None):  # noqa: E501
        """Secret - a model defined in Swagger

        :param name: The name of this Secret.  # noqa: E501
        :type name: str
        :param value: The value of this Secret.  # noqa: E501
        :type value: str
        """
        self.swagger_types = {
            'name': str,
            'value': str
        }

        self.attribute_map = {
            'name': 'name',
            'value': 'value'
        }

        self._name = name
        self._value = value

    @classmethod
    def from_dict(cls, dikt) -> 'Secret':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Secret of this Secret.  # noqa: E501
        :rtype: Secret
        """
        return util.deserialize_model(dikt, cls)

    @property
    def name(self) -> str:
        """Gets the name of this Secret.

        Name of secret  # noqa: E501

        :return: The name of this Secret.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this Secret.

        Name of secret  # noqa: E501

        :param name: The name of this Secret.
        :type name: str
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

    @property
    def value(self) -> str:
        """Gets the value of this Secret.

        Value of secret in plain-text  # noqa: E501

        :return: The value of this Secret.
        :rtype: str
        """
        return self._value

    @value.setter
    def value(self, value: str):
        """Sets the value of this Secret.

        Value of secret in plain-text  # noqa: E501

        :param value: The value of this Secret.
        :type value: str
        """

        self._value = value
