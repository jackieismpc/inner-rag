import http from './index'
import { tokenStore } from './token'

// 账号没有自助注册：由管理员用 scripts/create_user.py 创建（见 README「账号与权限」）
export const authApi = {
  // 登录成功即落 token，调用方只需要处理「谁登录了」
  login: async (username, password) => {
    const res = await http.post('/auth/login', { username, password })
    tokenStore.set(res.data.access_token)
    return res.data.user
  },
  // 刷新页面后用 token 换回用户信息，token 失效会走 401 拦截
  me: () => http.get('/auth/me'),
}

export { tokenStore }
