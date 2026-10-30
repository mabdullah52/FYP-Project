"""Keeps the app's data (patients, studies, reports, patient-app data) in Azure Blob Storage.

The online server's own disk is wiped whenever Azure restarts or scales the app to zero, and
Azure Container Apps "express" cannot mount Azure Files. So the server works on its local disk
and this module copies that folder to a blob container:

* at start-up everything stored in the container is downloaded first (restore);
* afterwards, every few seconds after a change and once a minute anyway, new and changed files
  are uploaded and files that were deleted locally are deleted from the container (sync);
* SQLite databases are copied with SQLite's backup API, so a copy is never taken mid-write;
* a last sync runs when the server shuts down.

Enabled only when the environment variable AZURE_STORAGE_CONNECTION_STRING is set
(optional: LUNGSCAN_BLOB_CONTAINER, default "lungscan-data"). Without it nothing changes.
"""
import os
import sqlite3
import tempfile
import threading
import time

CONN = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
CONTAINER = os.environ.get("LUNGSCAN_BLOB_CONTAINER", "lungscan-data").strip() or "lungscan-data"
SKIP_SUFFIXES = (".db-journal", ".db-wal", ".db-shm", ".tmp", ".write_test")


class BlobMirror:
    def __init__(self, local_dir, conn=CONN, container=CONTAINER):
        from azure.storage.blob import BlobServiceClient
        self.local = os.path.abspath(local_dir)
        self.client = BlobServiceClient.from_connection_string(conn).get_container_client(container)
        try:
            self.client.create_container()
        except Exception:            # already exists
            pass
        self.seen = {}               # blob name -> (size, mtime) of the local file last uploaded
        self.remote = set()          # blob names that exist in the container
        self.dirty = threading.Event()
        self.lock = threading.Lock()
        self.stop = threading.Event()

    # ------------------------------------------------------------------ helpers
    def _name(self, path):
        return os.path.relpath(path, self.local).replace(os.sep, "/")

    def _local_files(self):
        for root, _, files in os.walk(self.local):
            for f in files:
                if not f.endswith(SKIP_SUFFIXES):
                    yield os.path.join(root, f)

    def _upload(self, path, name):
        if path.endswith(".db"):     # consistent copy of a live SQLite database
            fd, tmp = tempfile.mkstemp(suffix=".db")
            os.close(fd)
            try:
                src, dst = sqlite3.connect(path), sqlite3.connect(tmp)
                with dst:
                    src.backup(dst)
                src.close(); dst.close()
                with open(tmp, "rb") as fh:
                    self.client.upload_blob(name, fh, overwrite=True)
            finally:
                os.remove(tmp)
        else:
            with open(path, "rb") as fh:
                self.client.upload_blob(name, fh, overwrite=True)

    # ------------------------------------------------------------------ restore + sync
    def restore(self):
        os.makedirs(self.local, exist_ok=True)
        n = 0
        for blob in self.client.list_blobs():
            self.remote.add(blob.name)
            dest = os.path.join(self.local, *blob.name.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                self.client.download_blob(blob.name).readinto(fh)
            st = os.stat(dest)
            self.seen[blob.name] = (st.st_size, st.st_mtime)
            n += 1
        return n

    def sync(self):
        with self.lock:
            up, gone = 0, 0
            present = set()
            for path in self._local_files():
                name = self._name(path)
                present.add(name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                sig = (st.st_size, st.st_mtime)
                if self.seen.get(name) == sig and name in self.remote:
                    continue
                if time.time() - st.st_mtime < 2 and not name.endswith(".db"):
                    self.dirty.set()     # still being written: take it next round
                    continue
                try:
                    self._upload(path, name)
                except Exception as e:   # keep going; it is retried next round
                    print("storage sync: upload failed for", name, "-", e)
                    continue
                self.seen[name] = sig
                self.remote.add(name)
                up += 1
            for name in sorted(self.remote - present):
                try:
                    self.client.delete_blob(name)
                except Exception:
                    pass
                self.remote.discard(name)
                self.seen.pop(name, None)
                gone += 1
            return up, gone

    def run_forever(self, every_s=60, settle_s=3):
        last = 0.0
        while not self.stop.is_set():
            if self.dirty.wait(timeout=5):
                time.sleep(settle_s)       # let a burst of writes finish
                self.dirty.clear()
            elif time.time() - last < every_s:
                continue
            try:
                self.sync()
            except Exception as e:
                print("storage sync failed:", e)
            last = time.time()

    def start(self):
        threading.Thread(target=self.run_forever, daemon=True).start()


def mirror_for(local_dir):
    """A started BlobMirror (after restoring the data), or None when no storage is configured."""
    if not CONN:
        return None
    m = BlobMirror(local_dir)
    n = m.restore()
    print(f"storage: restored {n} files from blob container '{CONTAINER}'")
    m.start()
    return m
