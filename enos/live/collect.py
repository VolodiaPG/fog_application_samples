import csv
import multiprocessing as mp
import os
import sys
import tarfile
import tempfile
from datetime import datetime
from io import TextIOWrapper

import requests
from integration import aliases


def worker(queue, url, metrixName):
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        with TextIOWrapper(tmpfile, encoding="utf-8") as file:
            writer = csv.writer(file, delimiter="\t")
            response = requests.get(
                "{0}/api/v1/query".format(url),
                params={"query": metrixName + f'[{os.environ["period"]}]'},
            )
            results = response.json()["data"]["result"]
            # Build a list of all labelnames used.
            # gets all keys and discard __name__
            labelnames = set()
            for result in results:
                labelnames.update(result["metric"].keys())
            # Canonicalize
            labelnames.discard("__name__")
            labelnames = sorted(labelnames)

            writer.writerow(["name"] + labelnames + ["timestamp", "value"])
            for result in results:
                for label in labelnames:
                    ll = list(result["metric"].values())
                    for value in result["values"]:
                        writer.writerow(ll + value)
        tmpfile.close()  # Close before sending to threads
        queue.put((metrixName, tmpfile.name))


def names(queue):
    names = aliases()
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        with TextIOWrapper(tmpfile, encoding="utf-8") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(["instance", "name"])
            for (key, value) in names.items():
                writer.writerow([key, value])
        tmpfile.close()  # Close before sending to threads
        queue.put(("names", tmpfile.name))


def listener(queue, filepath):
    with tarfile.open(filepath, "w:xz", preset=9) as tar:
        while True:
            m = queue.get()
            if m == "kill":
                break

            metrixName, tmpfile_name = m

            tarinfo = tarfile.TarInfo(f"{metrixName}.csv")
            tarinfo.size = os.path.getsize(tmpfile_name)
            tarinfo.mtime = os.path.getmtime(tmpfile_name)
            with open(tmpfile_name, "rb") as tmpfile:
                tar.addfile(tarinfo, tmpfile)

            os.remove(tmpfile_name)


if __name__ == "__main__":
    """
    Prometheus hourly data as csv.
    """
    URL = f"http://localhost:{os.environ['port']}"

    def GetMetrixNames(url):
        response = requests.get("{0}/api/v1/label/__name__/values".format(url))
        names = response.json()["data"]  # Return metrix names
        return names

    if len(sys.argv) != 1:
        print(
            f"Usage: {sys.argv[0]}\n use port env variable to set the port on localhost"
        )
        sys.exit(1)
    metrixNames = GetMetrixNames(URL)

    today = datetime.today()
    today = today.strftime("%Y-%m-%d-%H-%M")
    prefix_dir = "metrics-arks"
    prefix_filename = os.getenv("DEPLOYMENT_NAME")
    if prefix_filename is None:
        prefix_filename = ""
    try:
        os.mkdir(prefix_dir)
    except FileExistsError:
        pass
    archive = f"{prefix_dir}/metrics_{prefix_filename}_{today}.tar.xz"

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    # put listener to work first
    pool.apply_async(listener, (queue, archive))
    pool.apply_async(names, (queue,))

    jobs = []
    for metrixName in metrixNames:
        job = pool.apply_async(worker, (queue, URL, metrixName))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        if job is not None:
            job.get()

    queue.put("kill")
    pool.close()
    pool.join()

    print(f"Finished writing archive {archive}")

    try:
        os.remove("latest_metrics.tar.xz")
    except (FileExistsError, FileNotFoundError):
        pass
    os.symlink(archive, "latest_metrics.tar.xz")
