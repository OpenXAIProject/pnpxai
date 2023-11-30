import time

def serve():
    for i in range(10):
        print('server is still alive')
        time.sleep(1)

if __name__ == '__main__':
    serve()