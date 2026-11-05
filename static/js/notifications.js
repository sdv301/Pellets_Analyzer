/**
 * Pellets Analyzer — Notifications Center (Phase 1)
 * Manages notification bell, unread counter, dropdown list and mark-as-read
 */

(function () {
  'use strict';

  const STORAGE_KEY = 'pellets_read_notifications';

  function getReadIds() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch (e) {
      return [];
    }
  }

  function setReadIds(ids) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
    } catch (e) {
      console.warn('Could not save read notifications', e);
    }
  }

  async function loadNotifications() {
    const listContainer = document.getElementById('notificationsList');
    const badge = document.getElementById('notificationsBadge');
    if (!listContainer || !badge) return;

    try {
      const response = await fetch('/api/notifications');
      if (!response.ok) return;
      const data = await response.json();
      if (!data.success || !Array.isArray(data.notifications)) return;

      const notifications = data.notifications;
      const readIds = getReadIds();
      const unreadCount = notifications.filter(n => !readIds.includes(n.id)).length;

      // Update badge
      if (unreadCount > 0) {
        badge.textContent = unreadCount > 9 ? '9+' : unreadCount;
        badge.classList.remove('d-none');
      } else {
        badge.classList.add('d-none');
      }

      // Render list
      listContainer.innerHTML = '';
      if (notifications.length === 0) {
        listContainer.innerHTML = `
          <div class="p-3 text-center text-muted text-xs">
            <i class="fas fa-bell-slash text-sm d-block mb-1"></i>
            Нет новых уведомлений
          </div>
        `;
        return;
      }

      notifications.forEach(item => {
        const isRead = readIds.includes(item.id);
        const itemEl = document.createElement('div');
        itemEl.className = `notification-item ${isRead ? 'read' : 'unread'}`;
        
        const iconClass = item.category === 'success' 
          ? 'fa-circle-check text-success' 
          : (item.category === 'warning' ? 'fa-triangle-exclamation text-warning' : 'fa-info-circle text-primary');

        itemEl.innerHTML = `
          <div class="mt-1"><i class="fas ${iconClass}"></i></div>
          <div class="flex-grow-1">
            <div class="d-flex align-items-center justify-content-between">
              <span class="text-xs fw-bold text-dark">${item.title}</span>
              <small class="text-muted text-xxs">${item.time || ''}</small>
            </div>
            <p class="text-xs text-muted mb-0 mt-0.5">${item.message}</p>
          </div>
        `;
        listContainer.appendChild(itemEl);
      });
    } catch (e) {
      console.warn('Failed to load notifications', e);
    }
  }

  function markAllRead() {
    const listContainer = document.getElementById('notificationsList');
    const badge = document.getElementById('notificationsBadge');
    
    fetch('/api/notifications')
      .then(res => res.json())
      .then(data => {
        if (data.success && Array.isArray(data.notifications)) {
          const allIds = data.notifications.map(n => n.id);
          setReadIds(allIds);
          if (badge) badge.classList.add('d-none');
          if (listContainer) {
            listContainer.querySelectorAll('.notification-item').forEach(el => {
              el.classList.remove('unread');
              el.classList.add('read');
            });
          }
        }
      })
      .catch(e => console.warn('Could not mark all as read', e));
  }

  document.addEventListener('DOMContentLoaded', function () {
    loadNotifications();

    const markAllBtn = document.getElementById('markAllNotificationsRead');
    if (markAllBtn) {
      markAllBtn.addEventListener('click', function (e) {
        e.preventDefault();
        markAllRead();
      });
    }
  });
})();
