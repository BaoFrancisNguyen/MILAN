/**
 * FAUCON - Dashboard.js
 * Fonctionnalités pour le dashboard interactif
 */

// Configuration du dashboard
let dashboardConfig = {
  layout: [],
  charts: {}
};

// État du dashboard
let isDragging = false;
let currentDragElement = null;

/**
 * Initialise le dashboard interactif
 */
function initDashboard() {
  // Récupérer le conteneur du dashboard
  const dashboardContainer = document.getElementById('dashboard-container');
  if (!dashboardContainer) return;
  
  // Charger la configuration sauvegardée si disponible
  const savedConfig = localStorage.getItem('dashboard-config');
  if (savedConfig) {
    try {
      dashboardConfig = JSON.parse(savedConfig);
      renderDashboard();
    } catch (e) {
      console.error('Erreur lors du chargement de la configuration du dashboard:', e);
      // Réinitialiser avec une configuration par défaut
      initDefaultDashboard();
    }
  } else {
    // Aucune configuration existante, initialiser par défaut
    initDefaultDashboard();
  }
  
  // Configurer le bouton d'ajout de graphique
  const addChartBtn = document.getElementById('add-chart-btn');
  if (addChartBtn) {
    addChartBtn.addEventListener('click', showAddChartModal);
  }
  
  // Configurer le bouton de sauvegarde du dashboard
  const saveDashboardBtn = document.getElementById('save-dashboard-btn');
  if (saveDashboardBtn) {
    saveDashboardBtn.addEventListener('click', saveDashboardConfig);
  }
  
  // Configurer le bouton de réinitialisation du dashboard
  const resetDashboardBtn = document.getElementById('reset-dashboard-btn');
  if (resetDashboardBtn) {
    resetDashboardBtn.addEventListener('click', resetDashboard);
  }
  
  // Configurer le glisser-déposer pour les widgets
  setupDragAndDrop();
}

/**
 * Initialise le dashboard avec une configuration par défaut
 */
function initDefaultDashboard() {
  // Configuration par défaut avec 4 graphiques
  dashboardConfig = {
    layout: [
      {id: 'chart-1', x: 0, y: 0, w: 6, h: 4, type: 'Histogramme'},
      {id: 'chart-2', x: 6, y: 0, w: 6, h: 4, type: 'Carte de chaleur'},
      {id: 'chart-3', x: 0, y: 4, w: 6, h: 4, type: 'Graphique en barres'},
      {id: 'chart-4', x: 6, y: 4, w: 6, h: 4, type: 'Camembert'}
    ],
    charts: {
      'chart-1': {
        type: 'Histogramme',
        title: 'Distribution',
        x_var: null,
        color_var: null
      },
      'chart-2': {
        type: 'Carte de chaleur',
        title: 'Corrélations',
        x_var: null,
        y_var: null
      },
      'chart-3': {
        type: 'Graphique en barres',
        title: 'Comparaison',
        x_var: null,
        y_var: null
      },
      'chart-4': {
        type: 'Camembert',
        title: 'Répartition',
        x_var: null
      }
    }
  };
  
  // Rendre le dashboard
  renderDashboard();
}

/**
 * Affiche la modale pour ajouter un nouveau graphique
 */
function showAddChartModal() {
  const modal = new bootstrap.Modal(document.getElementById('add-chart-modal'));
  modal.show();
  
  // Récupérer les colonnes disponibles
  fetchColumns()
    .then(columns => {
      // Mettre à jour les sélecteurs avec les colonnes disponibles
      const xVarSelect = document.getElementById('new-chart-x-var');
      const yVarSelect = document.getElementById('new-chart-y-var');
      const colorVarSelect = document.getElementById('new-chart-color-var');
      
      if (xVarSelect) updateSelectOptions(xVarSelect, columns);
      if (yVarSelect) updateSelectOptions(yVarSelect, columns);
      if (colorVarSelect) updateSelectOptions(colorVarSelect, columns);
    })
    .catch(error => {
      console.error('Erreur lors de la récupération des colonnes:', error);
    });
  
  // Configurer l'événement du type de graphique
  const chartTypeSelect = document.getElementById('new-chart-type');
  if (chartTypeSelect) {
    chartTypeSelect.addEventListener('change', function() {
      updateChartOptions(this.value);
    });
    // Déclencher le changement initial
    updateChartOptions(chartTypeSelect.value);
  }
  
  // Configurer le bouton d'ajout
  const addBtn = document.getElementById('add-chart-confirm-btn');
  if (addBtn) {
    addBtn.addEventListener('click', function() {
      addNewChart();
      modal.hide();
    });
  }
}

/**
 * Met à jour les options d'un sélecteur
 */
function updateSelectOptions(select, options) {
  // Conserver la valeur sélectionnée
  const selectedValue = select.value;
  
  // Vider le sélecteur
  select.innerHTML = '';
  
  // Ajouter l'option vide
  const emptyOption = document.createElement('option');
  emptyOption.value = '';
  emptyOption.textContent = '-- Sélectionner --';
  select.appendChild(emptyOption);
  
  // Ajouter les options des colonnes
  options.forEach(column => {
    const option = document.createElement('option');
    option.value = column;
    option.textContent = column;
    select.appendChild(option);
  });
  
  // Restaurer la valeur sélectionnée si elle existe
  if (selectedValue && options.includes(selectedValue)) {
    select.value = selectedValue;
  }
}

/**
 * Met à jour les options du formulaire en fonction du type de graphique
 */
function updateChartOptions(chartType) {
  const yVarGroup = document.getElementById('new-chart-y-var-group');
  const colorVarGroup = document.getElementById('new-chart-color-var-group');
  
  // Réinitialiser la visibilité
  if (yVarGroup) yVarGroup.style.display = 'block';
  if (colorVarGroup) colorVarGroup.style.display = 'block';
  
  // Ajuster en fonction du type de graphique
  switch (chartType) {
    case 'Camembert':
      if (yVarGroup) yVarGroup.style.display = 'none';
      break;
    case 'Histogramme':
      if (yVarGroup) yVarGroup.style.display = 'none';
      break;
  }
}

/**
 * Ajoute un nouveau graphique au dashboard
 */
function addNewChart() {
  // Récupérer les valeurs du formulaire
  const chartType = document.getElementById('new-chart-type').value;
  const chartTitle = document.getElementById('new-chart-title').value || 'Nouveau graphique';
  const xVar = document.getElementById('new-chart-x-var').value;
  const yVar = document.getElementById('new-chart-y-var').value;
  const colorVar = document.getElementById('new-chart-color-var').value;
  
  // Générer un ID unique
  const chartId = `chart-${Date.now()}`;
  
  // Ajouter à la configuration du dashboard
  dashboardConfig.layout.push({
    id: chartId,
    x: 0, // Sera positionné lors du rendu
    y: 0,
    w: 6,
    h: 4,
    type: chartType
  });
  
  dashboardConfig.charts[chartId] = {
    type: chartType,
    title: chartTitle,
    x_var: xVar,
    y_var: yVar,
    color_var: colorVar
  };
  
  // Réorganiser la disposition
  reorganizeLayout();
  
  // Rendre le dashboard
  renderDashboard();
  
  // Sauvegarder la configuration
  saveDashboardConfig();
}

/**
 * Récupère les colonnes disponibles du dataset actuel
 */
async function fetchColumns() {
  try {
    const response = await fetch('/api/columns');
    const data = await response.json();
    
    if (data.error) {
      console.error('Erreur API:', data.error);
      return [];
    }
    
    return data.columns || [];
  } catch (error) {
    console.error('Erreur lors de la récupération des colonnes:', error);
    return [];
  }
}

/**
 * Réorganise la disposition du dashboard
 */
function reorganizeLayout() {
  const gridSize = 12; // Largeur totale de la grille (Bootstrap utilise 12 colonnes)
  const itemsPerRow = 2; // Nombre d'éléments par ligne
  const itemWidth = gridSize / itemsPerRow; // Largeur de chaque élément
  const itemHeight = 4; // Hauteur standard
  
  dashboardConfig.layout.forEach((item, index) => {
    const row = Math.floor(index / itemsPerRow);
    const col = index % itemsPerRow;
    
    item.x = col * itemWidth;
    item.y = row * itemHeight;
    item.w = itemWidth;
    item.h = itemHeight;
  });
}

/**
 * Rend le dashboard avec la configuration actuelle
 */
function renderDashboard() {
  const dashboardContainer = document.getElementById('dashboard-container');
  if (!dashboardContainer) return;
  
  // Vider le conteneur
  dashboardContainer.innerHTML = '';
  
  // Créer les widgets pour chaque graphique dans la disposition
  dashboardConfig.layout.forEach(item => {
    const chartConfig = dashboardConfig.charts[item.id];
    if (!chartConfig) return;
    
    // Créer le widget
    const widget = document.createElement('div');
    widget.className = 'dashboard-widget';
    widget.id = `widget-${item.id}`;
    widget.setAttribute('data-chart-id', item.id);
    widget.style.gridColumn = `span ${item.w}`;
    widget.style.gridRow = `span ${item.h}`;
    
    // Structure interne du widget
    widget.innerHTML = `
      <div class="card h-100">
        <div class="card-header d-flex justify-content-between align-items-center">
          <div class="d-flex align-items-center">
            <span class="drag-handle me-2"><i class="bi bi-grip-vertical"></i></span>
            <h5 class="mb-0">${chartConfig.title || chartConfig.type}</h5>
          </div>
          <div class="widget-actions">
            <button class="btn btn-sm btn-outline-secondary widget-edit-btn" data-chart-id="${item.id}">
              <i class="bi bi-pencil"></i>
            </button>
            <button class="btn btn-sm btn-outline-danger widget-remove-btn" data-chart-id="${item.id}">
              <i class="bi bi-x"></i>
            </button>
          </div>
        </div>
        <div class="card-body">
          <div id="${item.id}" class="chart-container" style="width:100%; height:100%;"></div>
        </div>
      </div>
    `;
    
    // Ajouter au conteneur
    dashboardContainer.appendChild(widget);
    
    // Créer le graphique
    createDashboardChart(item.id, chartConfig);
  });
  
  // Configurer les actions des widgets
  setupWidgetActions();
}

/**
 * Crée un graphique pour le dashboard
 */
function createDashboardChart(chartId, config) {
  // Vérifier si les données requises sont disponibles
  if (!config.x_var && config.type !== 'Carte de chaleur') {
    // Afficher un message demandant de configurer le graphique
    const chartContainer = document.getElementById(chartId);
    if (chartContainer) {
      chartContainer.innerHTML = `
        <div class="d-flex flex-column justify-content-center align-items-center h-100">
          <i class="bi bi-gear-fill fs-1 text-muted mb-2"></i>
          <p class="text-muted text-center">Configurez ce graphique en cliquant sur <i class="bi bi-pencil"></i></p>
        </div>
      `;
    }
    return;
  }
  
  // Préparer les paramètres de la requête
  const params = {
    chart_type: config.type,
    x_var: config.x_var
  };
  
  if (config.y_var) params.y_var = config.y_var;
  if (config.color_var) params.color_var = config.color_var;
  
  // Afficher un indicateur de chargement
  const chartContainer = document.getElementById(chartId);
  if (!chartContainer) return;
  
  chartContainer.innerHTML = '<div class="d-flex justify-content-center align-items-center h-100"><div class="spinner-border text-primary" role="status"></div></div>';
  
  // Appeler l'API pour créer la visualisation
  fetch('/api/create-visualization', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      chartContainer.innerHTML = `
        <div class="alert alert-danger m-3">
          <i class="bi bi-exclamation-triangle-fill me-2"></i>
          ${data.error}
        </div>
      `;
      return;
    }
    
    // Nettoyer le conteneur
    chartContainer.innerHTML = '';
    
    // Créer le graphique avec Plotly
    Plotly.newPlot(chartId, data.data, data.layout, {
      responsive: true,
      displayModeBar: false
    });
  })
  .catch(error => {
    console.error('Erreur lors de la création du graphique:', error);
    chartContainer.innerHTML = `
      <div class="alert alert-danger m-3">
        <i class="bi bi-exclamation-triangle-fill me-2"></i>
        Erreur de communication avec le serveur
      </div>
    `;
  });
}

/**
 * Configure les actions pour les widgets (édition, suppression)
 */
function setupWidgetActions() {
  // Boutons d'édition
  document.querySelectorAll('.widget-edit-btn').forEach(button => {
    button.addEventListener('click', function() {
      const chartId = this.getAttribute('data-chart-id');
      editChart(chartId);
    });
  });
  
  // Boutons de suppression
  document.querySelectorAll('.widget-remove-btn').forEach(button => {
    button.addEventListener('click', function() {
      const chartId = this.getAttribute('data-chart-id');
      removeChart(chartId);
    });
  });
}

/**
 * Affiche la modale d'édition pour un graphique
 */
function editChart(chartId) {
  const chartConfig = dashboardConfig.charts[chartId];
  if (!chartConfig) return;
  
  // Récupérer la modale
  const modal = new bootstrap.Modal(document.getElementById('edit-chart-modal'));
  
  // Remplir le formulaire avec les valeurs actuelles
  document.getElementById('edit-chart-id').value = chartId;
  document.getElementById('edit-chart-title').value = chartConfig.title || '';
  
  const chartTypeSelect = document.getElementById('edit-chart-type');
  if (chartTypeSelect) {
    // Sélectionner le type actuel
    for (let i = 0; i < chartTypeSelect.options.length; i++) {
      if (chartTypeSelect.options[i].value === chartConfig.type) {
        chartTypeSelect.selectedIndex = i;
        break;
      }
    }
    
    // Mettre à jour les options en fonction du type
    updateEditChartOptions(chartConfig.type);
    
    // Configurer l'événement de changement
    chartTypeSelect.addEventListener('change', function() {
      updateEditChartOptions(this.value);
    });
  }
  
  // Récupérer les colonnes et remplir les sélecteurs
  fetchColumns()
    .then(columns => {
      const xVarSelect = document.getElementById('edit-chart-x-var');
      const yVarSelect = document.getElementById('edit-chart-y-var');
      const colorVarSelect = document.getElementById('edit-chart-color-var');
      
      // Mettre à jour les options
      if (xVarSelect) {
        updateSelectOptions(xVarSelect, columns);
        xVarSelect.value = chartConfig.x_var || '';
      }
      
      if (yVarSelect) {
        updateSelectOptions(yVarSelect, columns);
        yVarSelect.value = chartConfig.y_var || '';
      }
      
      if (colorVarSelect) {
        updateSelectOptions(colorVarSelect, columns);
        colorVarSelect.value = chartConfig.color_var || '';
      }
    })
    .catch(error => {
      console.error('Erreur lors de la récupération des colonnes:', error);
    });
  
  // Configurer le bouton de sauvegarde
  const saveBtn = document.getElementById('edit-chart-save-btn');
  if (saveBtn) {
    saveBtn.onclick = function() {
      saveChartChanges();
      modal.hide();
    };
  }
  
  // Afficher la modale
  modal.show();
}

/**
 * Met à jour les options du formulaire d'édition
 */
function updateEditChartOptions(chartType) {
  const yVarGroup = document.getElementById('edit-chart-y-var-group');
  const colorVarGroup = document.getElementById('edit-chart-color-var-group');
  
  // Réinitialiser la visibilité
  if (yVarGroup) yVarGroup.style.display = 'block';
  if (colorVarGroup) colorVarGroup.style.display = 'block';
  
  // Ajuster en fonction du type de graphique
  switch (chartType) {
    case 'Camembert':
      if (yVarGroup) yVarGroup.style.display = 'none';
      break;
    case 'Histogramme':
      if (yVarGroup) yVarGroup.style.display = 'none';
      break;
  }
}

/**
 * Sauvegarde les modifications d'un graphique
 */
function saveChartChanges() {
  const chartId = document.getElementById('edit-chart-id').value;
  if (!chartId || !dashboardConfig.charts[chartId]) return;
  
  // Récupérer les nouvelles valeurs
  const chartTitle = document.getElementById('edit-chart-title').value || 'Graphique';
  const chartType = document.getElementById('edit-chart-type').value;
  const xVar = document.getElementById('edit-chart-x-var').value;
  const yVar = document.getElementById('edit-chart-y-var').value;
  const colorVar = document.getElementById('edit-chart-color-var').value;
  
  // Mettre à jour la configuration
  dashboardConfig.charts[chartId] = {
    ...dashboardConfig.charts[chartId],
    title: chartTitle,
    type: chartType,
    x_var: xVar,
    y_var: yVar,
    color_var: colorVar
  };
  
  // Mettre à jour le type dans la disposition
  const layoutItem = dashboardConfig.layout.find(item => item.id === chartId);
  if (layoutItem) {
    layoutItem.type = chartType;
  }
  
  // Mettre à jour le titre dans le DOM
  const widgetHeader = document.querySelector(`#widget-${chartId} .card-header h5`);
  if (widgetHeader) {
    widgetHeader.textContent = chartTitle;
  }
  
  // Recréer le graphique
  createDashboardChart(chartId, dashboardConfig.charts[chartId]);
  
  // Sauvegarder la configuration
  saveDashboardConfig();
}

/**
 * Supprime un graphique du dashboard
 */
function removeChart(chartId) {
  // Demander confirmation
  if (!confirm('Êtes-vous sûr de vouloir supprimer ce graphique ?')) return;
  
  // Supprimer de la configuration
  dashboardConfig.layout = dashboardConfig.layout.filter(item => item.id !== chartId);
  delete dashboardConfig.charts[chartId];
  
  // Réorganiser la disposition
  reorganizeLayout();
  
  // Rendre le dashboard
  renderDashboard();
  
  // Sauvegarder la configuration
  saveDashboardConfig();
}

/**
 * Sauvegarde la configuration du dashboard
 */
function saveDashboardConfig() {
  localStorage.setItem('dashboard-config', JSON.stringify(dashboardConfig));
}

/**
 * Réinitialise le dashboard à sa configuration par défaut
 */
function resetDashboard() {
  // Demander confirmation
  if (!confirm('Êtes-vous sûr de vouloir réinitialiser le dashboard ? Toutes vos personnalisations seront perdues.')) return;
  
  // Initialiser avec la configuration par défaut
  initDefaultDashboard();
  
  // Sauvegarder la configuration
  saveDashboardConfig();
}

/**
 * Configure le glisser-déposer pour les widgets
 */
function setupDragAndDrop() {
  // Implémenter avec une bibliothèque comme Sortable.js
  // Pour garder cet exemple simple, nous ne l'implémentons pas complètement ici
  
  // Code basique pour illustrer l'idée :
  document.querySelectorAll('.drag-handle').forEach(handle => {
    handle.addEventListener('mousedown', function(e) {
      const widget = this.closest('.dashboard-widget');
      if (!widget) return;
      
      isDragging = true;
      currentDragElement = widget;
      
      // Logique de glisser-déposer...
    });
  });
}

// Initialiser le dashboard au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
  // Vérifier si le dashboard doit être initialisé
  if (document.getElementById('dashboard-container')) {
    initDashboard();
  }
});
