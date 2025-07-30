// Dashboard JavaScript for Redisense Energy Monitoring

// Global variables
let autoRefreshInterval;
let autoRefreshEnabled = false;
let charts = {};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

function initializeDashboard() {
    // Update timestamp
    updateTimestamp();

    // Initialize tooltips
    initializeTooltips();

    // Initialize animations
    initializeAnimations();

    // Initialize responsive behavior
    initializeResponsiveBehavior();

    // Initialize keyboard shortcuts
    initializeKeyboardShortcuts();

    // Check connection status
    checkConnectionStatus();

    // Initialize counter animations
    animateCounters();
}

// Counter animation
function animateCounters() {
    const counters = document.querySelectorAll('.counter');
    const animationDuration = 2000; // 2 seconds

    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target')) || parseInt(counter.textContent);
        const suffix = counter.getAttribute('data-suffix') || '';
        let current = 0;
        const increment = target / (animationDuration / 50);

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            counter.textContent = Math.floor(current) + suffix;
        }, 50);
    });
}

// Smooth animations
function initializeAnimations() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in-up');
    });

    // Add loading animation to buttons on click
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            this.classList.add('loading');
            setTimeout(() => {
                this.classList.remove('loading');
            }, 1000);
        });
    });
}

// Timestamp functions
function updateTimestamp() {
    const now = new Date();
    const timestampElement = document.getElementById('timestamp');
    if (timestampElement) {
        timestampElement.textContent = now.toLocaleTimeString();
    }
}

// Auto-refresh functionality with better UX
function toggleAutoRefresh() {
    const icon = document.getElementById('auto-refresh-icon');

    if (autoRefreshEnabled) {
        clearInterval(autoRefreshInterval);
        autoRefreshEnabled = false;
        if (icon) {
            icon.className = 'bi bi-play-fill';
        }
        showNotification('Auto-refresh disabled', 'info');
    } else {
        autoRefreshInterval = setInterval(() => {
            refreshMetrics();
            updateTimestamp();
        }, 30000); // 30 seconds
        autoRefreshEnabled = true;
        if (icon) {
            icon.className = 'bi bi-pause-fill';
        }
        showNotification('Auto-refresh enabled (30s interval)', 'success');
    }
}

// Enhanced refresh functions
function refreshDashboard() {
    showLoadingState();

    // Add smooth transition
    document.body.style.opacity = '0.8';

    setTimeout(() => {
        location.reload();
    }, 500);
}

function showLoadingState() {
    const cards = document.querySelectorAll('.metric-card');
    cards.forEach(card => {
        card.classList.add('loading');
    });
}

function refreshMetrics() {
    showLoadingState();

    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => {
            updateMetrics(data);
            updateTimestamp();
            showNotification('Metrics updated', 'success');
        })
        .catch(error => {
            console.error('Error refreshing metrics:', error);
            showNotification('Error refreshing metrics', 'error');
        })
        .finally(() => {
            hideLoadingState();
        });
}

function hideLoadingState() {
    const cards = document.querySelectorAll('.metric-card');
    cards.forEach(card => {
        card.classList.remove('loading');
    });
}

function updateMetrics(data) {
    // Update metric values with smooth animations
    const elements = {
        'total-devices': data.total_devices,
        'current-power': data.current_power + ' kW',
        'avg-consumption': data.avg_consumption + ' kW',
        'total-energy': data.total_energy_24h + ' kWh',
        'active-devices': data.active_devices,
        'offline-devices': data.offline_devices
    };

    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            // Animate the update with a subtle pulse
            element.style.transform = 'scale(1.1)';
            element.style.transition = 'transform 0.3s ease';

            setTimeout(() => {
                element.textContent = value;
                element.style.transform = 'scale(1)';
            }, 150);
        }
    });
}

// Enhanced notification system
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notif => notif.remove());

    const notification = document.createElement('div');
    notification.className = `notification alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    notification.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="bi bi-${getNotificationIcon(type)} me-2"></i>
            <span>${message}</span>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    document.body.appendChild(notification);

    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 150);
        }
    }, 5000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'check-circle-fill';
        case 'error': return 'exclamation-triangle-fill';
        case 'warning': return 'exclamation-circle-fill';
        default: return 'info-circle-fill';
    }
}

// Connection status checking
function checkConnectionStatus() {
    const statusIndicator = document.getElementById('status-indicator');

    fetch('/api/health')
        .then(response => {
            if (response.ok) {
                if (statusIndicator) {
                    statusIndicator.innerHTML = '<i class="bi bi-circle-fill"></i> Online';
                    statusIndicator.className = 'badge bg-success status-online';
                }
            } else {
                throw new Error('Service unavailable');
            }
        })
        .catch(() => {
            if (statusIndicator) {
                statusIndicator.innerHTML = '<i class="bi bi-circle-fill"></i> Offline';
                statusIndicator.className = 'badge bg-danger status-offline';
            }
        });
}

// Initialize tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Responsive behavior
function initializeResponsiveBehavior() {
    // Handle sidebar collapse on mobile
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');

    if (navbarToggler && navbarCollapse) {
        navbarToggler.addEventListener('click', function() {
            setTimeout(() => {
                navbarCollapse.classList.toggle('show');
            }, 100);
        });
    }
}

// Keyboard shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + R for refresh
        if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
            e.preventDefault();
            refreshMetrics();
        }

        // Ctrl/Cmd + Space for auto-refresh toggle
        if ((e.ctrlKey || e.metaKey) && e.code === 'Space') {
            e.preventDefault();
            toggleAutoRefresh();
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            });
        }
    });
}

// Device management functions
function viewDevice(deviceId) {
    window.location.href = `/devices/${deviceId}`;
}

function editDevice(deviceId) {
    // Open edit modal or redirect to edit page
    showNotification(`Edit device ${deviceId}`, 'info');
}

function deleteDevice(deviceId) {
    if (confirm(`Are you sure you want to delete device ${deviceId}?`)) {
        fetch(`/api/devices/${deviceId}`, {
            method: 'DELETE'
        })
        .then(response => {
            if (response.ok) {
                showNotification('Device deleted successfully', 'success');
                setTimeout(() => location.reload(), 1000);
            } else {
                throw new Error('Failed to delete device');
            }
        })
        .catch(error => {
            showNotification('Error deleting device', 'error');
        });
    }
}

// Search functionality
function performSearch(query) {
    if (!query.trim()) return;

    const searchResults = document.getElementById('search-results');
    if (searchResults) {
        searchResults.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div></div>';
    }

    fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
            displaySearchResults(data.results);
        })
        .catch(error => {
            console.error('Search error:', error);
            showNotification('Search failed', 'error');
        });
}

function displaySearchResults(results) {
    const searchResults = document.getElementById('search-results');
    if (!searchResults) return;

    if (results.length === 0) {
        searchResults.innerHTML = '<div class="alert alert-info">No results found</div>';
        return;
    }

    let html = '<div class="row g-3">';
    results.forEach(result => {
        html += `
            <div class="col-md-6">
                <div class="card search-result-card">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <h6 class="card-title">${result.device_id}</h6>
                            <span class="search-score">${(result.score * 100).toFixed(1)}%</span>
                        </div>
                        <p class="card-text small text-muted">${result.content}</p>
                        <button class="btn btn-sm btn-primary" onclick="viewDevice('${result.device_id}')">
                            View Details
                        </button>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';

    searchResults.innerHTML = html;
}

// Chart initialization and management
function initializeChart(elementId, chartType, data, options = {}) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return null;

    // Destroy existing chart if it exists
    if (charts[elementId]) {
        charts[elementId].destroy();
    }

    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top'
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            },
            x: {
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            }
        }
    };

    const chartConfig = {
        type: chartType,
        data: data,
        options: { ...defaultOptions, ...options }
    };

    charts[elementId] = new Chart(ctx, chartConfig);
    return charts[elementId];
}

// Utility functions
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatDate(date) {
    return new Date(date).toLocaleDateString();
}

function formatTime(date) {
    return new Date(date).toLocaleTimeString();
}

// Export functions for global access
window.refreshDashboard = refreshDashboard;
window.toggleAutoRefresh = toggleAutoRefresh;
window.refreshMetrics = refreshMetrics;
window.viewDevice = viewDevice;
window.editDevice = editDevice;
window.deleteDevice = deleteDevice;
window.performSearch = performSearch;
                element.textContent = value;
                element.style.transform = 'scale(1)';
            }, 150);
        }
    });

    // Update status indicator
    updateStatusIndicator(data.active_devices, data.offline_devices);
}

// Status indicator
function updateStatusIndicator(active, offline) {
    const statusElement = document.getElementById('status-indicator');
    if (!statusElement) return;

    const total = active + offline;
    if (total === 0) {
        statusElement.innerHTML = '<i class="bi bi-circle-fill"></i> No Data';
        statusElement.className = 'badge bg-secondary';
    } else if (offline === 0) {
        statusElement.innerHTML = '<i class="bi bi-circle-fill"></i> All Online';
        statusElement.className = 'badge bg-success';
    } else if (offline < total / 2) {
        statusElement.innerHTML = '<i class="bi bi-circle-fill"></i> Mostly Online';
        statusElement.className = 'badge bg-warning';
    } else {
        statusElement.innerHTML = '<i class="bi bi-circle-fill"></i> Issues Detected';
        statusElement.className = 'badge bg-danger';
    }
}

// Connection status check
function checkConnectionStatus() {
    fetch('/api/metrics')
        .then(response => {
            if (response.ok) {
                updateConnectionStatus(true);
            } else {
                updateConnectionStatus(false);
            }
        })
        .catch(error => {
            updateConnectionStatus(false);
        });
}

function updateConnectionStatus(isOnline) {
    const statusElement = document.getElementById('status-indicator');
    if (!statusElement) return;

    if (isOnline) {
        statusElement.innerHTML = '<i class="bi bi-circle-fill"></i> Online';
        statusElement.className = 'badge bg-success';
    } else {
        statusElement.innerHTML = '<i class="bi bi-circle-fill"></i> Offline';
        statusElement.className = 'badge bg-danger';
    }
}

// Notification system
function showNotification(message, type = 'info', duration = 3000) {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification-toast');
    existingNotifications.forEach(notification => notification.remove());

    // Create notification
    const notification = document.createElement('div');
    notification.className = `notification-toast alert alert-${type} position-fixed`;
    notification.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        opacity: 0;
        transform: translateX(100%);
        transition: all 0.3s ease;
    `;

    const icon = {
        'success': 'check-circle-fill',
        'error': 'x-circle-fill',
        'warning': 'exclamation-triangle-fill',
        'info': 'info-circle-fill'
    }[type] || 'info-circle-fill';

    notification.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="bi bi-${icon} me-2"></i>
            <span>${message}</span>
            <button type="button" class="btn-close ms-auto" onclick="this.parentElement.parentElement.remove()"></button>
        </div>
    `;

    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
    }, 100);

    // Auto remove
    if (duration > 0) {
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }
}

// Loading state
function showLoadingState() {
    const loadingOverlay = document.createElement('div');
    loadingOverlay.id = 'loading-overlay';
    loadingOverlay.className = 'position-fixed top-0 start-0 w-100 h-100';
    loadingOverlay.style.cssText = `
        background: rgba(255, 255, 255, 0.8);
        z-index: 9998;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    loadingOverlay.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div class="h5 text-primary">Loading...</div>
        </div>
    `;

    document.body.appendChild(loadingOverlay);
}

function hideLoadingState() {
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.remove();
    }
}

// Chart utilities
function createGradient(ctx, color1, color2) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
}

function formatChartData(data, labelKey, valueKey) {
    return {
        labels: data.map(item => item[labelKey]),
        datasets: [{
            data: data.map(item => item[valueKey]),
            backgroundColor: [
                '#0d6efd', '#198754', '#ffc107', '#dc3545', '#6f42c1',
                '#fd7e14', '#20c997', '#6c757d', '#e83e8c', '#17a2b8'
            ]
        }]
    };
}

// Search functionality
function performSearch(query, limit = 10) {
    if (!query.trim()) {
        showNotification('Please enter a search query', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('query', query);

    showLoadingState();

    fetch('/api/search', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingState();
        displaySearchResults(data);
    })
    .catch(error => {
        hideLoadingState();
        console.error('Search error:', error);
        showNotification('Search error: ' + error.message, 'error');
    });
}

function displaySearchResults(results) {
    // This would be implemented in the search page
    console.log('Search results:', results);
}

// Device utilities
function getDeviceStatusClass(status) {
    const statusClasses = {
        'normal': 'text-success',
        'maintenance': 'text-warning',
        'anomaly': 'text-danger',
        'offline': 'text-secondary'
    };
    return statusClasses[status] || 'text-secondary';
}

function getDeviceStatusIcon(status) {
    const statusIcons = {
        'normal': 'check-circle-fill',
        'maintenance': 'exclamation-triangle-fill',
        'anomaly': 'x-circle-fill',
        'offline': 'circle'
    };
    return statusIcons[status] || 'circle';
}

// Tooltip initialization
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Responsive behavior
function initializeResponsiveBehavior() {
    // Handle window resize
    window.addEventListener('resize', function() {
        // Update charts if they exist
        if (typeof Chart !== 'undefined' && Chart.instances.length > 0) {
            Chart.instances.forEach(chart => {
                chart.resize();
            });
        }
    });

    // Handle mobile navigation
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');

    if (navbarToggler && navbarCollapse) {
        navbarToggler.addEventListener('click', function() {
            navbarCollapse.classList.toggle('show');
        });
    }
}

// Keyboard shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + R: Refresh
        if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
            event.preventDefault();
            refreshDashboard();
        }

        // Ctrl/Cmd + F: Focus search
        if ((event.ctrlKey || event.metaKey) && event.key === 'f') {
            const searchInput = document.querySelector('input[name="q"]');
            if (searchInput) {
                event.preventDefault();
                searchInput.focus();
            }
        }

        // Escape: Close modals
        if (event.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            });
        }
    });
}

// Export functionality
function exportData(format = 'csv') {
    showNotification(`Exporting data as ${format.toUpperCase()}...`, 'info');

    // In a real implementation, this would generate and download a file
    setTimeout(() => {
        showNotification(`${format.toUpperCase()} export would be generated here`, 'success');
    }, 1000);
}

// Utility functions
function formatNumber(number, decimals = 2) {
    return Number(number).toFixed(decimals);
}

function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Error handling
window.addEventListener('error', function(event) {
    console.error('JavaScript error:', event.error);
    showNotification('An error occurred. Please refresh the page.', 'error', 5000);
});

// Fetch error handler
function handleFetchError(response) {
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response;
}

// Initialize periodic checks
setInterval(checkConnectionStatus, 60000); // Check every minute
setInterval(updateTimestamp, 1000); // Update timestamp every second
