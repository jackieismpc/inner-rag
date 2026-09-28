<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 px-4">
    <div class="w-full max-w-sm">
      <div class="flex items-center justify-center gap-2.5 mb-6">
        <div class="w-9 h-9 rounded-xl bg-primary-600 flex items-center justify-center">
          <svg class="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
        </div>
        <span class="font-semibold text-gray-900">ModelX RAG</span>
      </div>

      <div class="card p-6">
        <h1 class="text-base font-semibold text-gray-900 mb-1">登录</h1>
        <p class="text-xs text-gray-500 mb-5">登录后只能看到自己拥有或被授权的知识库</p>

        <form class="space-y-4" @submit.prevent="onSubmit">
          <div>
            <label class="text-xs font-medium text-gray-700 mb-1.5 block">用户名</label>
            <input v-model.trim="form.username" required autocomplete="username" class="input" placeholder="用户名" />
          </div>
          <div>
            <label class="text-xs font-medium text-gray-700 mb-1.5 block">密码</label>
            <input v-model="form.password" type="password" required autocomplete="current-password" class="input" placeholder="密码" />
          </div>

          <p v-if="error" class="text-xs text-red-500">{{ error }}</p>

          <button type="submit" class="btn-primary w-full justify-center" :disabled="submitting">
            <Spinner v-if="submitting" class="w-3.5 h-3.5" />
            登录
          </button>
        </form>
      </div>

      <p class="text-xs text-gray-400 text-center mt-4">
        账号由管理员创建（<code>scripts/create_user.py</code>），系统不开放自助注册
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
