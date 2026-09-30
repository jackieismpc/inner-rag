<template>
  <div class="min-h-screen flex items-center justify-center bg-neutral-50 px-4 relative overflow-hidden">
    <!-- 背景柔光装饰 -->
    <div class="absolute -top-32 -left-32 w-96 h-96 rounded-full bg-primary-100/50 blur-3xl"></div>
    <div class="absolute -bottom-32 -right-32 w-96 h-96 rounded-full bg-primary-200/40 blur-3xl"></div>

    <div class="w-full max-w-sm relative">
      <div class="flex items-center justify-center gap-3 mb-8">
        <div class="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-primary-glow">
          <svg class="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
        </div>
        <div class="text-left">
          <p class="text-xl font-semibold text-neutral-900 tracking-tight">ModelX RAG</p>
          <p class="text-xs text-neutral-500">企业知识库 · 智能问答</p>
        </div>
      </div>

      <div class="card p-7 shadow-soft">
        <h1 class="text-lg font-semibold text-neutral-900 mb-1">欢迎回来</h1>
        <p class="text-xs text-neutral-500 mb-6">登录后只能看到自己拥有或被授权的知识库</p>

        <form class="space-y-4" @submit.prevent="onSubmit">
          <div>
            <label class="text-xs font-medium text-neutral-700 mb-1.5 block">用户名</label>
            <input v-model.trim="form.username" required autocomplete="username" class="input" placeholder="请输入用户名" />
          </div>
          <div>
            <label class="text-xs font-medium text-neutral-700 mb-1.5 block">密码</label>
            <input v-model="form.password" type="password" required autocomplete="current-password" class="input" placeholder="请输入密码" />
          </div>

          <p v-if="error" class="text-xs text-danger">{{ error }}</p>

          <button type="submit" class="btn-primary w-full justify-center h-10" :disabled="submitting">
            <Spinner v-if="submitting" class="w-4 h-4" />
            登录
          </button>
        </form>
      </div>

      <p class="text-xs text-neutral-400 text-center mt-5">
        账号由管理员创建（<code class="bg-neutral-100 px-1.5 py-0.5 rounded text-neutral-500 font-mono">scripts/create_user.py</code>），系统不开放自助注册
      </p>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAppStore } from '@/stores/app'
import Spinner from '@/components/Spinner.vue'

const appStore = useAppStore()
const router = useRouter()
const route = useRoute()

const form = reactive({ username: '', password: '' })
const error = ref('')
const submitting = ref(false)

async function onSubmit() {
  submitting.value = true
  error.value = ''
  try {
    await appStore.login(form.username, form.password)
    // 回到被拦下的那个页面，而不是一律丢回首页
    router.replace(typeof route.query.redirect === 'string' ? route.query.redirect : '/kb')
  } catch (e) {
    error.value = e.message
  } finally {
    submitting.value = false
  }
}
</script>
