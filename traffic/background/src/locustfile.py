import random
import string
import time
from typing import Dict

from locust import HttpUser, between, task, constant_pacing
from requests.exceptions import RequestException

USER_AGENTS = [
    # Desktop Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/124.0 Safari/537.36",

    # Firefox
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",

    # Mobile Safari
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) "
    "AppleWebKit/605.1.15 Version/17.0 Mobile Safari/604.1",

    # Android Chrome
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
    "AppleWebKit/537.36 Chrome/124.0 Mobile Safari/537.36",
]

ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "pl-PL,pl;q=0.9,en;q=0.8",
    "de-DE,de;q=0.9,en;q=0.8",
]

STREAM_CHUNK_SIZES = [
    1024,
    2048,
    4096,
    8192,
    16384,
]

def random_text(min_len=20, max_len=5000):
    size = random.randint(min_len, max_len)
    return ''.join(
        random.choices(
            string.ascii_letters + string.digits + "     ",
            k=size
        )
    )


def random_headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": random.choice(ACCEPT_LANGUAGES),
        "Connection": random.choice(["keep-alive", "close"]),
        "Cache-Control": random.choice([
            "no-cache",
            "max-age=0",
        ]),
    }


def partial_stream_consume(
    response,
    min_chunks=2,
    max_chunks=50,
    sleep_probability=0.2,
):
    limit = random.randint(min_chunks, max_chunks)
    chunk_size = random.choice(STREAM_CHUNK_SIZES)

    for i, _ in enumerate(response.iter_content(chunk_size=chunk_size)):
        if i >= limit:
            break

        # simulate reading latency / network jitter
        if random.random() < sleep_probability:
            time.sleep(random.uniform(0.01, 0.2))


def safe_request(fn, retries=2):
    for attempt in range(retries + 1):
        try:
            return fn()
        except RequestException:
            if attempt >= retries:
                raise
            time.sleep(random.uniform(0.2, 1.0))

class BaseWebUser(HttpUser):
    abstract = True

    network_timeout = 30

    def on_start(self):
        self.client.headers.update(random_headers())

        # Initial landing page
        self.safe_get("/")

    def safe_get(self, path, **kwargs):
        return safe_request(
            lambda: self.client.get(
                path,
                timeout=self.network_timeout,
                **kwargs
            )
        )

    def safe_post(self, path, **kwargs):
        return safe_request(
            lambda: self.client.post(
                path,
                timeout=self.network_timeout,
                **kwargs
            )
        )

    def submit_random_form(self, min_size=20, max_size=2000):
        payload = {
            "data": random_text(min_size, max_size)
        }

        with self.client.post(
            "/rudy",
            data=payload,
            catch_response=True,
            timeout=self.network_timeout,
        ) as response:

            if response.status_code >= 500:
                response.failure("Server error on /rudy")
            else:
                response.success()

    def consume_download_stream(self, aggressive=False):
        max_chunks = 120 if aggressive else 30

        with self.client.get(
            "/download",
            stream=True,
            catch_response=True,
            timeout=60,
        ) as response:

            try:
                partial_stream_consume(
                    response,
                    min_chunks=3,
                    max_chunks=max_chunks,
                )
                response.success()

            except Exception as exc:
                response.failure(str(exc))

    def consume_slow_stream(self, long_read=False):
        max_chunks = 250 if long_read else 40

        with self.client.get(
            "/slow-read",
            stream=True,
            catch_response=True,
            timeout=120,
        ) as response:

            try:
                partial_stream_consume(
                    response,
                    min_chunks=5,
                    max_chunks=max_chunks,
                    sleep_probability=0.35,
                )

                response.success()

            except Exception as exc:
                response.failure(str(exc))

class PassiveUser(BaseWebUser):
    wait_time = between(5, 20)
    weight = 5

    @task(8)
    def browse_home(self):
        self.safe_get("/")

    @task(1)
    def occasional_form(self):
        self.submit_random_form(20, 200)

    @task(1)
    def leave_quickly(self):
        self.environment.runner.quit if random.random() < 0.0001 else None

class ActiveUser(BaseWebUser):
    wait_time = between(2, 8)
    weight = 4

    @task(5)
    def browse(self):
        self.safe_get("/")

    @task(3)
    def submit_medium_form(self):
        self.submit_random_form(100, 3000)

    @task(1)
    def read_stream(self):
        self.consume_slow_stream()

    @task(1)
    def partial_download(self):
        self.consume_download_stream()

class DownloadFocusedUser(BaseWebUser):
    wait_time = between(10, 30)
    weight = 1

    @task(2)
    def browse_before_download(self):
        self.safe_get("/")

    @task(3)
    def large_download(self):
        self.consume_download_stream(aggressive=True)

    @task(1)
    def long_stream_read(self):
        self.consume_slow_stream(long_read=True)


class MobileUser(BaseWebUser):
    wait_time = between(4, 15)
    weight = 3

    def on_start(self):
        super().on_start()

        # force mobile UA
        self.client.headers["User-Agent"] = random.choice([
            ua for ua in USER_AGENTS
            if "Mobile" in ua or "iPhone" in ua
        ])

    @task(6)
    def light_browsing(self):
        self.safe_get("/")

    @task(2)
    def small_form(self):
        self.submit_random_form(20, 400)

    @task(1)
    def interrupted_stream(self):
        self.consume_slow_stream(long_read=False)

class BounceUser(BaseWebUser):
    wait_time = between(0.5, 2)
    weight = 2

    @task
    def bounce(self):
        self.safe_get("/")

        # many real users leave immediately
        if random.random() < 0.8:
            self.stop(True)


class PowerUser(BaseWebUser):
    wait_time = constant_pacing(5)
    weight = 1

    @task(5)
    def browse(self):
        self.safe_get("/")

    @task(4)
    def heavy_forms(self):
        self.submit_random_form(500, 10000)

    @task(2)
    def stream_reader(self):
        self.consume_slow_stream(long_read=True)

    @task(2)
    def download(self):
        self.consume_download_stream(aggressive=True)