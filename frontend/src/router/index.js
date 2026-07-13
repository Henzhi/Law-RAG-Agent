import { createRouter, createWebHashHistory } from 'vue-router'
import LoginView from '../views/LoginView.vue'
import ChatView from '../views/ChatView.vue'

const routes = [
  { path: '/', name: 'chat', component: ChatView },
  { path: '/login', name: 'login', component: LoginView },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

export default router
