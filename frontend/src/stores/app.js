import { defineStore } from 'pinia'
import { ref } from 'vue'
import { authApi, tokenStore } from '@/api/auth'

export const useAppStore = defineStore('app', () => {
  const systemOk = ref(null)
  const toast = ref(null)
  // 当前登录用户；null 表示未登录（token 是否还有效由后端 /auth/me 说了算）
  const user = ref(null)
  let toastTimer = null

  function showToast(msg, type = 'info', duration = 3000) {
    if (toastTimer) clearTimeout(toastTimer)
    toast.value = { msg, type }
    toastTimer = setTimeout(() => toast.value = null, duration)
  }

  async function login(username, password) {
    user.value = await authApi.login(username, password)
  }

  function logout() {
    tokenStore.clear()
    user.value = null
  }

  // 刷新页面后用 token 换回用户信息；token 失效会走 401 拦截回到登录页
  async function restoreSession() {
    if (!tokenStore.get()) return
    try {
      user.value = (await authApi.me()).data
    } catch {
      user.value = null
    }
  }

  return { systemOk, toast, user, showToast, login, logout, restoreSession }
})
