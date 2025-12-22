/**
 * A simple IndexedDB-backed cache for Transformers.js
 * Implements the match/put interface required by env.customCache
 */
export class IndexedDBCache {
  private dbName = "transformers-cache-db";
  private storeName = "transformers-cache";
  private db: IDBDatabase | null = null;

  private async getDB(): Promise<IDBDatabase> {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName);
        }
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db!);
      };

      request.onerror = () => {
        reject(new Error("Failed to open IndexedDB"));
      };
    });
  }

  async match(request: Request | string): Promise<Response | undefined> {
    const url = typeof request === "string" ? request : request.url;
    const db = await this.getDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readonly");
      const store = transaction.objectStore(this.storeName);
      const getRequest = store.get(url);

      getRequest.onsuccess = async () => {
        const data = getRequest.result;
        if (!data) {
          resolve(undefined);
          return;
        }

        // Reconstruct Response from stored Blob/ArrayBuffer
        const response = new Response(data.body, {
          headers: new Headers(data.headers),
          status: 200,
          statusText: "OK",
        });
        resolve(response);
      };

      getRequest.onerror = () => {
        reject(new Error("Failed to get from IndexedDB"));
      };
    });
  }

  async put(request: Request | string, response: Response): Promise<void> {
    const url = typeof request === "string" ? request : request.url;
    const db = await this.getDB();

    // We must clone and read the response body as a Blob to store it
    const blob = await response.clone().blob();
    const headers: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      headers[key] = value;
    });

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readwrite");
      const store = transaction.objectStore(this.storeName);
      
      const putRequest = store.put({
        url,
        body: blob,
        headers,
        timestamp: Date.now()
      }, url);

      putRequest.onsuccess = () => {
        resolve();
      };

      putRequest.onerror = () => {
        reject(new Error("Failed to put into IndexedDB"));
      };
    });
  }

  async clear(): Promise<void> {
    const db = await this.getDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readwrite");
      const store = transaction.objectStore(this.storeName);
      const request = store.clear();

      request.onsuccess = () => {
        this.db = null; // Close connection after clear if needed? No, just resolve.
        resolve();
      };

      request.onerror = () => {
        reject(new Error("Failed to clear IndexedDB"));
      };
    });
  }
}
