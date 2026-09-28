import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import { setUnauthorizedHandler } from './api'
import { useAppStore } from './stores/app'
import './assets/main.css'

const app = createApp(App)
app.use(createPinia())
app.use(router)

const appStore = useAppStore()

// 401 的统一出口（见 api/index.js）：token 过期/账号被停用后清登录态并回登录页。
// 已经在登录页就不再打断——「密码错误」也是 401，它只需要把错误文案显示在表单里。
setUnauthorizedHandler(() => {
  appStore.logout()
  if (!router.currentRoute.value.meta.public) router.replace({ path: '/login' })
})

app.mount('#app')
