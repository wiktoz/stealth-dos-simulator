from locust import HttpUser, task, between, SequentialTaskSet
import random
import string


def random_text(min_len=20, max_len=200):
    size = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))


class CasualBrowsing(SequentialTaskSet):

    def on_start(self):
        self.client.get("/")

    @task
    def browse(self):
        self.client.get("/")

    @task(2)
    def idle(self):
        pass


class CasualUser(HttpUser):
    tasks = [CasualBrowsing]
    wait_time = between(3, 10)
    weight = 5


class InteractiveSession(SequentialTaskSet):

    def on_start(self):
        self.client.get("/")

    @task(3)
    def browse(self):
        self.client.get("/")

    @task(2)
    def submit_form(self):
        payload = {"data": random_text(50, 300)}
        self.client.post("/rudy", data=payload)

    @task(1)
    def think(self):
        pass


class InteractiveUser(HttpUser):
    tasks = [InteractiveSession]
    wait_time = between(2, 6)
    weight = 3


class DownloaderBehavior(SequentialTaskSet):

    @task(2)
    def browse_before_download(self):
        self.client.get("/")

    @task(1)
    def download(self):
        # simulate partial consumption (common in real users)
        with self.client.get("/download", stream=True, catch_response=True) as response:
            for i, chunk in enumerate(response.iter_content(chunk_size=4096)):
                if i > random.randint(5, 20):
                    break


class DownloaderUser(HttpUser):
    tasks = [DownloaderBehavior]
    wait_time = between(5, 15)
    weight = 1


class ImpatientBehavior(SequentialTaskSet):

    @task(4)
    def rapid_browse(self):
        self.client.get("/")

    @task(2)
    def small_form(self):
        payload = {"data": random_text(10, 200)}
        self.client.post("/rudy", data=payload)

    @task(1)
    def occasional_download(self):
        with self.client.get("/download", stream=True, catch_response=True) as response:
            for i, _ in enumerate(response.iter_content(chunk_size=1024)):
                if i > 3:
                    break


class ImpatientUser(HttpUser):
    tasks = [ImpatientBehavior]
    wait_time = between(0.5, 2)
    weight = 1