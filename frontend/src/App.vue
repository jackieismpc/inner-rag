<template>
  <!-- 登录页不套应用外壳：未登录时不该看到侧边栏与系统状态 -->
  <router-view v-if="isPublicPage" />

  <div v-else class="flex h-screen overflow-hidden bg-neutral-50">
    <!-- Sidebar：毛玻璃材质 -->
    <aside class="w-60 flex-shrink-0 flex flex-col glass border-r border-neutral-200/60">
      <!-- Logo -->
      <div class="h-16 flex items-center gap-2.5 px-5">
        <div class="w-8 h-8 rounded-[10px] bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-primary-glow">
          <svg class="w-4.5 h-4.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
        </div>
        <div class="leading-tight">
          <p class="font-semibold text-[15px] text-neutral-800 tracking-tight">ModelX RAG</p>
          <p class="text-[11px] text-neutral-400">企业知识库 · 智能问答</p>
        </div>
      </div>

      <!-- Nav -->
      <nav class="flex-1 px-3 py-3 space-y-1">
        <p class="px-3 pb-1 text-[11px] font-medium uppercase tracking-wider text-neutral-400">工作台</p>
        <router-link to="/kb" class="nav-item" :class="{'nav-active': $route.path.startsWith('/kb')}">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
          知识库
        </router-link>
      </nav>

      <!-- 当前登录用户 -->
      <div class="px-3 pb-2">
        <div class="flex items-center gap-2.5 px-2 py-2 rounded-xl hover:bg-neutral-100/70 transition-colors">
          <span class="w-8 h-8 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 text-white flex items-center justify-center text-sm font-medium flex-shrink-0">
            {{ initial }}
          </span>
          <div class="flex-1 min-w-0">
            <p class="text-[13px] font-medium text-neutral-800 truncate">
              {{ appStore.user?.display_name || appStore.user?.username || '未登录' }}
            </p>
            <p class="text-[11px] text-neutral-400 truncate">{{ appStore.user?.username }}</p>
          </div>
          <button class="btn-ghost p-1.5" title="退出登录" @click="onLogout">
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/>
            </svg>
          </button>
        </div>
      </div>

      <!-- System status -->
      <div class="px-3 pb-4">
        <div class="rounded-xl bg-neutral-100/70 px-3 py-2.5">
          <div class="flex items-center gap-2 text-[12px] text-neutral-600">
            <span class="w-1.5 h-1.5 rounded-full flex-shrink-0"
              :class="health?.llm?.ok ? 'bg-success' : 'bg-neutral-300'"></span>
            <span :title="health?.llm?.error || ''">{{ llmLabel }}</span>
          </div>
          <div class="pl-3.5 pt-1 text-[11px] text-neutral-400 truncate" v-if="health?.llm?.model">
            {{ health.llm.model }}
          </div>
          <div class="pl-3.5 pt-0.5 text-[11px] text-neutral-400 truncate" v-if="health?.embedding?.model">
            embed · {{ health.embedding.model }}
          </div>
        </div>
      </div>
    </aside>

    <!-- Main content -->
    <main class="flex-1 overflow-hidden flex flex-col min-w-0">
      <router-view />
    </main>
  </div>

  <!-- Toast -->
  <transition name="fade">
    <div v-if="appStore.toast" class="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 rounded-2xl shadow-float text-sm font-medium"
      :class="{
        'bg-neutral-800/90 backdrop-blur text-white': appStore.toast.type === 'info',
        'bg-success text-white': appStore.toast.type === 'success',
        'bg-danger text-white':   appStore.toast.type === 'error',
      }">
      <span>{{ appStore.toast.msg }}</span>
    </div>
  </transition>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAppStore } from '@/stores/app'
import { sysApi } from '@/api'

const appStore = useAppStore()
const route = useRoute()
const router = useRouter()
const health = ref(null)

// 登录页不套应用外壳；系统状态接口（/api/system/health）本身免鉴权，所以登录页也能探活
const isPublicPage = computed(() => route.meta.public === true)
const initial = computed(() => (appStore.user?.username || '?').slice(0, 1).toUpperCase())

// /api/system/health 返回 { llm: { provider, model, ok, error }, embedding: {...} }
const llmLabel = computed(() => {
  const llm = health.value?.llm
  if (!llm) return '模型状态未知'
  return llm.ok ? `${llm.provider} 已连接` : `${llm.provider} 未连接`
})

function onLogout() {
  appStore.logout()
  appStore.showToast('已退出登录')
  router.replace('/login')
}

onMounted(async () => {
  // 刷新页面后用 token 换回登录用户；token 失效会被 401 拦截送回登录页
  await appStore.restoreSession()
  try { health.value = await sysApi.health() } catch {}
  setInterval(async () => {
    try { health.value = await sysApi.health() } catch {}
  }, 30000)
})
</script>

<style>
.nav-item {
  @apply flex items-center gap-2.5 px-3 py-2 rounded-xl text-[13px] text-neutral-600 font-medium
         hover:bg-neutral-100/70 hover:text-neutral-900 transition-colors cursor-pointer;
}
.nav-active {
  @apply bg-primary-50 text-primary-700;
}
</style>
