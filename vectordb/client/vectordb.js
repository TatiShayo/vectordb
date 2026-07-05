/**
 * VectorDB JavaScript/TypeScript client — H06
 * Works in Node.js and browser (fetch API).
 *
 * Usage (ESM):
 *   import { VectorDB } from './vectordb.js'
 *   const db = new VectorDB('http://localhost:8000', 'user-secret')
 *   const results = await db.search('products', queryVector, { topK: 10 })
 */

export class VectorDB {
  /**
   * @param {string} baseUrl  - Server URL
   * @param {string} apiKey   - API key
   * @param {object} opts     - { timeout: 30000 }
   */
  constructor(baseUrl = 'http://localhost:8000', apiKey = 'user-secret', opts = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '')
    this.headers = { 'Content-Type': 'application/json', 'X-API-Key': apiKey }
    this.timeout = opts.timeout ?? 30_000
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  async _fetch(method, path, body) {
    const ctrl = new AbortController()
    const tid = setTimeout(() => ctrl.abort(), this.timeout)
    try {
      const res = await fetch(this.baseUrl + path, {
        method, headers: this.headers, signal: ctrl.signal,
        body: body !== undefined ? JSON.stringify(body) : undefined,
      })
      clearTimeout(tid)
      if (res.status === 204) return null
      const data = await res.json()
      if (!res.ok) throw new Error(data?.detail ?? `HTTP ${res.status}`)
      return data
    } finally { clearTimeout(tid) }
  }

  _get(path)         { return this._fetch('GET', path) }
  _post(path, body)  { return this._fetch('POST', path, body) }
  _patch(path, body) { return this._fetch('PATCH', path, body) }
  _delete(path)      { return this._fetch('DELETE', path) }

  // ── Collections ───────────────────────────────────────────────────────────

  listCollections()           { return this._get('/collections') }
  getCollection(name)         { return this._get(`/collections/${name}`) }
  deleteCollection(name)      { return this._delete(`/collections/${name}`) }

  createCollection(name, { dimension = 384, distance = 'cosine',
    indexType = null, description = '' } = {}) {
    return this._post('/collections', {
      name, dimension, distance,
      ...(indexType ? { index_type: indexType } : {}),
      description,
    })
  }

  // ── Vectors ───────────────────────────────────────────────────────────────

  upsert(collection, id, vector, metadata = {}, { ttlSeconds } = {}) {
    return this._post(`/collections/${collection}/vectors`, {
      id, vector, metadata,
      ...(ttlSeconds ? { ttl_seconds: ttlSeconds } : {}),
    })
  }

  upsertBatch(collection, vectors) {
    // vectors: [{id, vector, metadata, ttlSeconds?}]
    return this._post(`/collections/${collection}/vectors/batch`, {
      vectors: vectors.map(v => ({
        id: v.id, vector: v.vector, metadata: v.metadata ?? {},
        ...(v.ttlSeconds ? { ttl_seconds: v.ttlSeconds } : {}),
      })),
    })
  }

  getVector(collection, id, { includeVector = false } = {}) {
    return this._get(`/collections/${collection}/vectors/${id}`
      + (includeVector ? '?include_vector=true' : ''))
  }

  deleteVector(collection, id) {
    return this._delete(`/collections/${collection}/vectors/${id}`)
  }

  patchMetadata(collection, id, metadata) {
    return this._patch(`/collections/${collection}/vectors/${id}`, { metadata })
  }

  scroll(collection, { limit = 100, offset = 0, includeVector = false } = {}) {
    return this._get(`/collections/${collection}/vectors/scroll`
      + `?limit=${limit}&offset=${offset}&include_vector=${includeVector}`)
  }

  count(collection) {
    return this._get(`/collections/${collection}/vectors/count`)
  }

  facets(collection, field, { limit = 100 } = {}) {
    return this._get(`/collections/${collection}/vectors/facets/${field}?limit=${limit}`)
  }

  // ── Search ────────────────────────────────────────────────────────────────

  search(collection, vector, {
    topK = 10, filter = null, includeVector = false,
    scoreThreshold = null, efSearch = null,
    useMMR = false, mmrLambda = 0.5,
    rerank = false, rerankQuery = null,
  } = {}) {
    const body = { vector, top_k: topK, include_vector: includeVector }
    if (filter)         body.filter = filter
    if (scoreThreshold) body.score_threshold = scoreThreshold
    if (efSearch)       body.ef_search = efSearch
    if (useMMR)         { body.use_mmr = true; body.mmr_lambda = mmrLambda }
    if (rerank)         { body.rerank = true; body.rerank_query = rerankQuery }
    return this._post(`/collections/${collection}/search`, body)
  }

  searchByText(collection, text, { topK = 10, filter = null, rerank = false } = {}) {
    return this._post(`/collections/${collection}/search/by-text`,
      { text, top_k: topK, ...(filter ? { filter } : {}), rerank })
  }

  searchById(collection, id, { topK = 10, filter = null } = {}) {
    return this._post(`/collections/${collection}/search/by-id`,
      { id, top_k: topK, ...(filter ? { filter } : {}) })
  }

  hybridSearch(collection, vector, text, {
    topK = 10, vectorWeight = 0.7, filter = null, fusion = 'rrf',
  } = {}) {
    return this._post(`/collections/${collection}/search/hybrid`, {
      vector, text, top_k: topK, vector_weight: vectorWeight,
      fusion, ...(filter ? { filter } : {}),
    })
  }

  batchSearch(collection, queries, { includeVector = false } = {}) {
    return this._post(`/collections/${collection}/search/batch`, {
      queries: queries.map(q => ({
        vector: q.vector, top_k: q.topK ?? 10,
        ...(q.filter ? { filter: q.filter } : {}),
      })),
      include_vector: includeVector,
    })
  }

  // ── Admin ─────────────────────────────────────────────────────────────────

  health()         { return this._get('/admin/health') }
  metrics()        { return this._get('/admin/metrics') }
  forceSave()      { return this._post('/admin/save', {}) }
  clearCache()     { return this._post('/admin/cache/clear', {}) }
  cacheStats()     { return this._get('/admin/cache/stats') }
  listTasks()      { return this._get('/admin/tasks') }
  getTask(id)      { return this._get(`/admin/tasks/${id}`) }

  rebuild(collection) {
    return this._post(`/admin/collections/${collection}/rebuild`, {})
  }

  createSnapshot(collection) {
    return this._post(`/admin/collections/${collection}/snapshot`, {})
  }
}

// CommonJS compat shim
if (typeof module !== 'undefined') module.exports = { VectorDB }
