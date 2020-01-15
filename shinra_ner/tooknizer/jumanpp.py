import argparse
import logging
import socketserver
import subprocess
from multiprocessing import Process


def main():
    logger = logging.getLogger(__name__)
    parser = _mk_argparser(logger=logger)
    args = parser.parse_arg()
    pass


def _mk_argparser(*, logger=None):
    logger = logger or logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Server mode of juman++")
    parser.add_argument('-c', '--cmd',
                        action='store',
                        default='jumanpp',
                        type=str,
                        help='which jumanpp cmd')
    parser.parser.add_argument('-p', '--port',
                               action='store',
                               default=12000,
                               type=int,
                               help='port number to open')
    return parser


class ExecuteDaemon:
    def __init__(self, cmdl, *, logger=None, **prs_args):
        self.cmdl = cmdl
        self.prs_args = prs_args
        self.logger = logger or logging.getLogger("ExecuteDaemon")
        self.start()

    def start(self):
        self.popen = subprocess.Popen(*self.cmdl,
                                      encoding='utf-8',
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      **self.prs_args)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.popen.stdin.close()

    def receive(self):
        self.popen.stdout.flush()
        return self.popen.stdout.read()


class JumanppRequestHandler(socketserver.StreamRequestHandler):
    def __init__(self, request, client_address, server, *, logger=None):
        self._logger = logging.getLogger('JumanppRequestHandler')
        self._logger.debug('__init__')
        super().__init__(request, client_address, server)

    def setup(self):
        self._logger.debug('setup')
        return super().setup()

    def handle(self):
        self._logger.debug('handle')
        data = self.rfile.readline()
        self._logger.debug('recv()->"%s"', data)
        self.request.send(data)

    def finish(self):
        self._logger.debug('finish')
        return super().finish()

    def _jumanpp_process():
        pass


class JumanppServer(socketserver.ForkingMixIn, socketserver.TCPServer):
    """
    Juman++のサーバとして動作するクラス
    """
    def __init__(self, server_address, handler_class, *, logger=None):
        self._logger = logger or logging.getLogger('JumnappServer')
        self._logger.debug('init')
        super().__init__(server_address, handler_class)
        self._logger.info('started jumanpp server')

    def server_activate(self):
        self._logger.info('server activated')
        super().server_activate()

    def serve_forever(self):
        self._logger.debug('waiting for request')
        self._logger.info('Handling requests, press <Ctrl-C> to quit')
        while True:
            self.handle_request()
        return

    def handle_request(self):
        self._logger.debug('handle_request')
        return super().handle_request()

    def verify_request(self, request, client_address):
        self._logger.debug('verify_request(%s, %s)', request, client_address)
        return super().verify_request(request, client_address)

    def process_request(self, request, client_address):
        self._logger.debug('process_request(%s, %s)', request, client_address)
        return super().process_request(request, client_address)

    def server_close(self):
        self._logger.debug('server_close')
        return super().server_close()

    def finish_request(self, request, client_address):
        self._logger.debug('finish_request(%s, %s)', request, client_address)
        return super().finish_request(request, client_address)

    def close_request(self, request_address):
        self._logger.debug('close_request(%s)', request_address)
        return super().close_request(request_address)


def test():
    import socket
    import threading
    import os

    address = ('localhost', 0)  # カーネルにポート番号を割り当てさせる
    server = JumanppServer(address, JumanppRequestHandler)
    ip, port = server.server_address  # 与えられたポート番号を調べる

    t = threading.Thread(target=server.serve_forever)
    t.setDaemon(True)  # 終了時にハングアップしない
    t.start()
    print('Server loop running in process:', os.getpid())

    # サーバへ接続する
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))

    # データを送る
    message = 'Hello, world'
    print('Sending : "%s"' % message)
    len_sent = s.send(message.encode())

    # レスポンスを受けとる
    response = s.recv(1024)
    print('Received: "%s"' % response)

    # クリーンアップ
    s.close()
    server.socket.close()
