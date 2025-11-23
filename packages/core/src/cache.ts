/**
 * Model caching strategy using Cache API
 */
export class ModelCache {
  private cacheName = "edge-llm-models";
  private cache: Cache | null = null;

  async initialize(): Promise<void> {
    if (typeof caches === "undefined") {
      console.warn("Cache API not available");
      return;
    }

    try {
      this.cache = await caches.open(this.cacheName);
    } catch (error) {
      console.warn("Failed to open cache:", error);
    }
  }

  /**
   * Store model data
   */
  async store(key: string, data: ArrayBuffer, ttl?: number): Promise<void> {
    if (!this.cache) {
      return;
    }

    try {
      const response = new Response(data, {
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Cache-Timestamp": Date.now().toString(),
          ...(ttl && { "X-Cache-TTL": ttl.toString() }),
        },
      });

      await this.cache.put(key, response);
    } catch (error) {
      console.warn(`Failed to store cache entry ${key}:`, error);
    }
  }

  /**
   * Retrieve model data
   */
  async retrieve(key: string): Promise<ArrayBuffer | null> {
    if (!this.cache) {
      return null;
    }

    try {
      const response = await this.cache.match(key);
      if (!response) {
        return null;
      }

      // Check TTL if present
      const ttlHeader = response.headers.get("X-Cache-TTL");
      const timestampHeader = response.headers.get("X-Cache-Timestamp");

      if (ttlHeader && timestampHeader) {
        const ttl = parseInt(ttlHeader, 10);
        const timestamp = parseInt(timestampHeader, 10);
        const now = Date.now();

        if (now - timestamp > ttl) {
          // Expired, delete and return null
          await this.cache.delete(key);
          return null;
        }
      }

      return await response.arrayBuffer();
    } catch (error) {
      console.warn(`Failed to retrieve cache entry ${key}:`, error);
      return null;
    }
  }

  /**
   * Check if key exists and is valid
   */
  async has(key: string): Promise<boolean> {
    if (!this.cache) {
      return false;
    }

    const data = await this.retrieve(key);
    return data !== null;
  }

  /**
   * Delete cache entry
   */
  async delete(key: string): Promise<void> {
    if (!this.cache) {
      return;
    }

    try {
      await this.cache.delete(key);
    } catch (error) {
      console.warn(`Failed to delete cache entry ${key}:`, error);
    }
  }

  /**
   * Clear all cached models
   */
  async clear(): Promise<void> {
    if (!this.cache) {
      return;
    }

    try {
      await caches.delete(this.cacheName);
      this.cache = null;
      await this.initialize();
    } catch (error) {
      console.warn("Failed to clear cache:", error);
    }
  }

  /**
   * Get cache size estimate
   */
  async getSize(): Promise<number> {
    if (!this.cache) {
      return 0;
    }

    try {
      const keys = await this.cache.keys();
      let totalSize = 0;

      for (const key of keys) {
        const response = await this.cache.match(key);
        if (response) {
          const blob = await response.blob();
          totalSize += blob.size;
        }
      }

      return totalSize;
    } catch (error) {
      console.warn("Failed to calculate cache size:", error);
      return 0;
    }
  }
}
