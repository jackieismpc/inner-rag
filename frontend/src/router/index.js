import { createRouter, createWebHistory } from 'vue-router'
import { tokenStore } from '@/api/token'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', component: () => import('@/views/LoginView.vue'), meta: { title: '登录', public: true } },
    { path: '/', redirect: '/kb' },
    { path: '/kb', component: () => import('@/views/KbList.vue'), meta: { title: '知识库' } },
    { path: '/kb/:id', component: () => import('@/views/KbDetail.vue'), meta: { title: '知识库详情' } },
    { path: '/kb/:id/chat', component: () => import('@/views/ChatView.vue'), meta: { title: '对话' } },
    { path: '/kb/:id/docs', component: () => import('@/views/DocList.vue'), meta: { title: '文档管理' } },
  ]
})

// 默认私有：新加页面忘了写 meta.public 时会被拦下，而不是裸奔。
// 只查本地有没有 token（同步、无闪屏），token 是否仍然有效由后端的 401 回答。
router.beforeEach((to) => {
  const authed = Boolean(tokenStore.get())
  if (to.meta.public) return authed ? { path: '/kb' } : true
  return authed ? true : { path: '/login', query: to.fullPath === '/' ? {} : { redirect: to.fullPath } }
})

export default router
