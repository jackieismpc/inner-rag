<template>
  <div class="flex flex-col h-full">
    <!-- Header -->
    <div class="flex-shrink-0 flex items-center gap-3 px-7 py-5">
      <button class="btn-ghost p-1.5" @click="$router.push('/kb')">
        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/>
        </svg>
      </button>
      <div class="w-10 h-10 rounded-2xl bg-primary-50 flex items-center justify-center text-xl flex-shrink-0">
        {{ kb?.icon || '📚' }}
      </div>
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <h1 class="text-lg font-bold text-neutral-900 truncate tracking-tight">{{ kb?.name }}</h1>
          <span v-if="kb && !isOwner" :class="permissionBadge(kb.my_permission)">
            {{ permissionLabel(kb.my_permission) }}
          </span>
        </div>
        <p class="text-[13px] text-neutral-500 truncate">{{ kb?.description || '暂无描述' }}</p>
      </div>
      <div class="flex gap-2">
        <!-- 成员管理仅拥有者可见：成员列表接口也要求 owner 权限 -->
        <button v-if="isOwner" class="btn-ghost text-xs gap-1.5" @click="openMembers">
          <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M17 20h5v-2a4 4 0 00-3-3.87M9 20H4v-2a4 4 0 013-3.87m6-1.13a4 4 0 10-4-4 4 4 0 004 4zm6-4a3 3 0 11-3-3"/>
          </svg>
          成员管理
        </button>
        <button class="btn-ghost text-xs gap-1.5" @click="$router.push(`/kb/${id}/docs`)">
          <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
          文档管理
        </button>
        <button class="btn-primary text-xs gap-1.5" @click="$router.push(`/kb/${id}/chat`)">
          <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>
          </svg>
          开始对话
        </button>
      </div>
    </div>

    <!-- Stats -->
    <div class="flex-1 overflow-auto p-7">
      <div class="grid grid-cols-3 gap-5 mb-7" v-if="kb">
        <div class="card p-5">
          <p class="text-[13px] text-neutral-500 mb-1.5">文档总数</p>
          <p class="text-3xl font-bold text-neutral-900 tracking-tight">{{ kb.doc_count }}</p>
        </div>
        <div class="card p-5">
          <p class="text-[13px] text-neutral-500 mb-1.5">向量数量</p>
          <p class="text-3xl font-bold text-primary-600 tracking-tight">{{ kb.vector_count ?? '-' }}</p>
        </div>
        <div class="card p-5">
          <p class="text-[13px] text-neutral-500 mb-1.5">嵌入模型</p>
          <p class="text-sm font-medium text-neutral-700 mt-2 truncate">{{ kb.embedding_model }}</p>
        </div>
      </div>

      <!-- Recent docs -->
      <div class="card overflow-hidden">
        <div class="flex items-center justify-between px-5 py-4 border-b border-neutral-100">
          <span class="text-[15px] font-semibold text-neutral-800">最近文档</span>
          <button class="text-[13px] font-medium text-primary-600 hover:text-primary-700" @click="$router.push(`/kb/${id}/docs`)">
            查看全部 →
          </button>
        </div>
        <div v-if="docs.length" class="divide-y divide-neutral-100">
          <div v-for="doc in docs" :key="doc.id" class="flex items-center gap-3 px-5 py-3.5 hover:bg-neutral-50 transition-colors">
            <span class="text-base flex-shrink-0">{{ fileIcon(doc.file_type) }}</span>
            <div class="flex-1 min-w-0">
              <p class="text-sm text-neutral-800 truncate">{{ doc.filename }}</p>
              <p class="text-xs text-neutral-400">{{ doc.chunk_count }} 块 · {{ formatSize(doc.file_size) }}</p>
            </div>
            <span :class="statusBadge(doc.status)">{{ statusLabel(doc.status) }}</span>
          </div>
        </div>
        <div v-else class="py-10 text-center text-sm text-neutral-400">暂无文档，去文档管理页上传</div>
      </div>
    </div>

    <!-- 成员授权（仅拥有者） -->
    <Modal v-if="showMembers" @close="showMembers = false">
      <template #title>成员授权</template>
      <div class="space-y-4">
        <div class="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50">
          <span class="badge-green">拥有者</span>
          <span class="text-sm text-gray-800">{{ members?.owner?.username }}</span>
        </div>

        <div v-if="members?.items?.length" class="rounded-lg border border-gray-100 divide-y divide-gray-50">
          <div v-for="m in members.items" :key="m.user_id" class="flex items-center gap-2 px-3 py-2">
            <span class="flex-1 text-sm text-gray-800 truncate">{{ m.username }}</span>
            <span :class="m.permission === 'write' ? 'badge-blue' : 'badge-gray'">
              {{ m.permission === 'write' ? '可写' : '只读' }}
            </span>
            <button class="btn-danger p-1" title="移除授权" @click="removeMember(m)">
              <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
              </svg>
            </button>
          </div>
        </div>
        <p v-else class="text-xs text-gray-400">除拥有者外暂无授权成员</p>

        <form class="flex gap-2" @submit.prevent="addMember">
          <input v-model.trim="memberForm.username" required class="input flex-1 text-sm" placeholder="用户名" />
          <select v-model="memberForm.permission" class="input w-24 text-sm">
            <option value="read">只读</option>
            <option value="write">可写</option>
          </select>
          <button type="submit" class="btn-primary text-xs" :disabled="memberBusy">授权</button>
        </form>
        <p class="text-xs text-gray-400">填登录名；重复授权同一用户即修改其权限。移除后立即失效。</p>
      </div>
    </Modal>
  </div>
</template>

<script setup>
import { computed, reactive, ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { kbApi, docApi } from '@/api'
import { useAppStore } from '@/stores/app'
import Modal from '@/components/Modal.vue'

const route = useRoute()
const appStore = useAppStore()
const id = parseInt(route.params.id)
const kb = ref(null)
const docs = ref([])

const isOwner = computed(() => kb.value?.my_permission === 'owner')

const showMembers = ref(false)
const members = ref(null)
const memberBusy = ref(false)
const memberForm = reactive({ username: '', permission: 'read' })

const fileIcon = t => ({ pdf:'📄', word:'📝', excel:'📊', text:'📃', image:'🖼️' }[t] || '📁')
const formatSize = b => b > 1048576 ? (b/1048576).toFixed(1)+'MB' : (b/1024).toFixed(1)+'KB'
const statusBadge = s => ({ completed:'badge-green', processing:'badge-blue', pending:'badge-yellow', failed:'badge-red' }[s] || 'badge-gray')
const statusLabel = s => ({ completed:'已完成', processing:'处理中', pending:'待处理', failed:'失败' }[s] || s)
const permissionBadge = p => ({ write:'badge-blue', read:'badge-gray' }[p] || 'badge-gray')
const permissionLabel = p => ({ owner:'拥有者', write:'可写', read:'只读' }[p] || '')

async function loadMembers() {
  try { members.value = (await kbApi.members(id)).data }
  catch (e) { appStore.showToast(e.message, 'error') }
}

function openMembers() {
  showMembers.value = true
  Object.assign(memberForm, { username: '', permission: 'read' })
  loadMembers()
}

async function addMember() {
  memberBusy.value = true
  try {
    await kbApi.addMember(id, memberForm)
    appStore.showToast('授权成功', 'success')
    memberForm.username = ''
    await loadMembers()
  } catch (e) { appStore.showToast(e.message, 'error') }
  finally { memberBusy.value = false }
}

async function removeMember(m) {
  try {
    await kbApi.removeMember(id, m.user_id)
    appStore.showToast('已移除授权', 'success')
    await loadMembers()
  } catch (e) { appStore.showToast(e.message, 'error') }
}

onMounted(async () => {
  try {
    kb.value = (await kbApi.get(id)).data
    const dres = await docApi.list({ kb_id: id, page: 1, page_size: 5 })
    docs.value = dres.data.items
  } catch (e) { appStore.showToast(e.message, 'error') }
})
</script>
