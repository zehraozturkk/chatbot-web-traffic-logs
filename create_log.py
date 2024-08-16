import random
from datetime import datetime, timedelta

# Örnek veriler
methods = ['GET', 'POST', 'PUT', 'DELETE']
statuses = [200, 301, 302, 404, 500]
urls = [
    '/home', '/product/123', '/product/456', '/checkout', '/category/electronics',
    '/cart', '/login', '/search', '/category/clothing', '/category/books'
]
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A5341f Safari/604.1',
    'Mozilla/5.0 (Linux; Android 10; SM-A205F Build/QP1A.190711.020) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Mobile Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36'
]
referers = ['http://google.com', 'http://facebook.com', 'http://mysite.com', '-', 'http://twitter.com']

# Zaman aralığı
start_time = datetime.now() - timedelta(days=1)
end_time = datetime.now()


def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

"""
IP Adresi: Rastgele bir IP adresi üretir.
Tarih ve Saat: Belirtilen zaman aralığında rastgele bir tarih ve saat üretir.
HTTP İsteği: GET, POST gibi bir HTTP isteği oluşturur.
HTTP Durum Kodu: Rastgele bir HTTP durum kodu seçilir.
Gönderilen Veri Boyutu: Rastgele bir veri boyutu (byte cinsinden) oluşturulur.
Yönlendiren (Referer): Rastgele bir yönlendiren URL'si veya boş değer seçilir.
Kullanıcı Aracısı (User-Agent): Farklı cihaz ve tarayıcılara ait rastgele bir kullanıcı ajanı seçilir.
"""

def generate_log_entry():
    ip = f'192.168.1.{random.randint(1, 255)}'
    date_time = random_date(start_time, end_time).strftime('%d/%b/%Y:%H:%M:%S +0300')
    method = random.choice(methods)
    url = random.choice(urls)
    status = random.choice(statuses)
    size = random.randint(1000, 10000)
    user_agent = random.choice(user_agents)
    referer = random.choice(referers)
    request = f'{method} {url} HTTP/1.1'

    # Nginx log formatına uygun string
    return f'{ip} - - [{date_time}] "{request}" {status} {size} "{referer}" "{user_agent}"'


# 200 log kaydı üret
log_entries = [generate_log_entry() for _ in range(200)]

# Logları dosyaya yaz
with open('nginx_logs.txt', 'w') as file:
    for entry in log_entries:
        file.write(entry + '\n')

print("Nginx log dosyası başarıyla oluşturuldu.")
