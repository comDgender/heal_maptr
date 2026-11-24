# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
import threading
from enum import Enum as _Enum
from socket import AF_INET, SO_REUSEADDR, SOCK_STREAM, SOL_SOCKET, socket
from typing import Any

__all__ = [
    "Enum",
    "SocketServer",
    "SocketClient",
    "unset_flag",
    "_UnsetFlag",
    "singleton",
]

logger = logging.getLogger(__file__)


class Enum(_Enum):
    @classmethod
    def value_of(cls, enum):
        if isinstance(enum, cls):
            pass
        elif isinstance(enum, str) and enum in cls._member_map_:
            enum = cls[enum]
        else:
            enum = cls(enum)
        return enum.value

    @classmethod
    def contains_value(cls, value):
        return value in cls._value2member_map_


def get_enum_value(x: Any) -> Any:
    """Turn an enum object into its value.

    If the parameter is not an enum object, return itself.
    """
    if isinstance(x, _Enum):
        return x.value
    return x


class SocketClient:
    def __init__(self, serve_address, serve_port):
        self.serve_address = serve_address
        self.serve_port = serve_port
        self.tcp_socket = socket(AF_INET, SOCK_STREAM)
        self.tcp_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

    def connect(self):
        self.tcp_socket.connect((self.serve_address, self.serve_port))

    def close(self):
        self.tcp_socket.close()

    def relaunch(self):
        self.close()
        self.tcp_socket = socket(AF_INET, SOCK_STREAM)
        self.tcp_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

    def send_msg(self, msg):
        assert isinstance(msg, str)
        self.tcp_socket.send(msg.encode())

    def receive_msg(self):
        from_serve_msg = self.tcp_socket.recv(1024).decode()
        return from_serve_msg


class SocketServer:
    def __init__(self, address=None, port=8012, num_listen=1):
        self.tcp_serve = socket(AF_INET, SOCK_STREAM)
        self.tcp_serve.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        if address is None:
            # using local ip
            address = ""
        self.tcp_serve.bind((address, port))
        self.tcp_serve.listen(num_listen)
        self.client_map = {}
        self.cur_client_address = None

    def wait_connect(self):
        client_socket, client_address = self.tcp_serve.accept()
        self.client_map[client_address] = client_socket
        self.cur_client_address = client_address

    def close(self):
        if self.cur_client_address in self.client_map:
            self.client_map[self.cur_client_address].close()

    def set_client(self, address):
        assert address in self.client_map
        self.cur_client_address = address

    def send_msg(self, msg):
        client_socket = self.client_map[self.cur_client_address]
        client_socket.send(msg.encode())

    def receive_msg(self):
        client_socket = self.client_map[self.cur_client_address]
        from_client_msg = client_socket.recv(1024).decode()
        return from_client_msg


class UnsetFlag:
    """Special flag for unset values."""


# for backward compatible
_UnsetFlag = UnsetFlag

unset_flag = UnsetFlag


def singleton(cls):
    _instance = {}
    _instance_lock = threading.Lock()

    def _singleton(*args, **kwargs):
        with _instance_lock:
            if cls not in _instance:
                _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton
