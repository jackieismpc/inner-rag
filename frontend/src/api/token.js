// Token 的唯一存取点。
// 单独成文件是为了避免循环依赖：请求拦截器（api/index.js）和登录态（api/auth.js）
// 都要用 token，如果放其中一边，两边就会互相 import。
const TOKEN_KEY = 'inner-rag.token'

export const tokenStore = {
  get: () => localStorage.getItem(TOKEN_KEY),
  set: (token) => localStorage.setItem(TOKEN_KEY, token),
  clear: () => localStorage.removeItem(TOKEN_KEY),
}
