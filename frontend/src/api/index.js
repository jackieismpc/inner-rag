import axios from 'axios'
import { tokenStore } from './token'

// 相对路径：走 Vite 代理（开发）或同源部署（生产）。
// 旧代码写死了绝对地址（且开头多一个空格），既绕过了代理，又与 vite.config.js 里的
// 端口配置重复，端口一改就全部失效。
const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '/api',
  timeout: 60000,
})

// 每个请求带上 Bearer token（登录接口自己不需要，带上也无害）
http.interceptors.request.use(config => {
  const token = tokenStore.get()
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// 401 的唯一处理点：token 过期/停用后清掉登录态并回到登录页。
// 注册回调而不是 import router：router 会 import 各视图、视图又 import 本文件，
// 直接引用会形成循环依赖。
let onUnauthorized = null
export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler
}

// 401 的唯一处理逻辑：清掉失效 token 并通知上层跳登录页。
// axios 拦截器与下面给 SSE 用的原生 fetch 都走这里，避免两套登录态逻辑。
export function handleUnauthorized() {
  tokenStore.clear()
  onUnauthorized?.()
}

// 原生 fetch（SSE 流式接口）用：axios 的拦截器管不到它，token 得自己带
// （axios 在浏览器里拿不到流式响应，所以 /chat/stream 只能用 fetch）
export function authHeaders() {
  const token = tokenStore.get()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

http.interceptors.response.use(
  res => res.data,
  err => {
    if (err.response?.status === 401) handleUnauthorized()
    const msg = err.response?.data?.detail || err.response?.data?.message || err.message
    return Promise.reject(new Error(msg))
  }
)

// Knowledge Base
export const kbApi = {
  list: (p) => http.get('/kb', { params: p }),
  get:  (id) => http.get(`/kb/${id}`),
  create: (d) => http.post('/kb', d),
  update: (id, d) => http.put(`/kb/${id}`, d),
  delete: (id) => http.delete(`/kb/${id}`),
  // 成员授权仅拥有者可用
  members: (id) => http.get(`/kb/${id}/members`),
  addMember: (id, d) => http.post(`/kb/${id}/members`, d),
  removeMember: (id, userId) => http.delete(`/kb/${id}/members/${userId}`),
}

// Document
export const docApi = {
  list:   (p) => http.get('/doc', { params: p }),
  get:    (id) => http.get(`/doc/${id}`),
  delete: (id) => http.delete(`/doc/${id}`),
  reprocess: (id) => http.post(`/doc/${id}/reprocess`),
  upload: (formData, onProgress) => http.post('/doc/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress && onProgress(Math.round(e.loaded / e.total * 100)),
  }),
  importPath: (d) => http.post('/doc/import-path', d),
}

// Chat
export const chatApi = {
  listConvs:  (p) => http.get('/chat/conversations', { params: p }),
  getMessages: (id) => http.get(`/chat/conversations/${id}/messages`),
  deleteConv: (id) => http.delete(`/chat/conversations/${id}`),
  send: (d) => http.post('/chat/send', d),
}

// System
export const sysApi = {
  health: () => http.get('/system/health'),
  models: () => http.get('/system/models'),
}

export default http
