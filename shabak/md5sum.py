import hashlib

with open('giphy.gif', 'rb') as gif:
    print(hashlib.md5(gif.read()).hexdigest())