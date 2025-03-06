/**
 * FAUCON - Visualisations Charts.js
 * Fonctionnalités pour la création et la gestion des visualisations
 */

// Stockage des graphiques actifs
const activeCharts = {};

/**
 * Initialise les visualisations sur la page
 */
function initVisualizations() {
  // Récupérer les conteneurs de visualisation
  const visualizationContainers = document.querySelectorAll('.visualization-container');
  
  // Si aucun conteneur, ne rien faire
  if (visualizationContainers.length === 0) return;
  
  // Configurer les contrôles pour chaque conteneur
  visualizationContainers.forEach((container, index) => {
    setupVisualizationControls(container, index);
  });
  
  // Configurer le sélecteur de type de graphique global
  const chartTypeSelector = document.getElementById('chart-type-selector');
  if (chartTypeSelector) {
    chartTypeSelector.addEventListener('change', function() {
      const selectedType = this.value;
      document.querySelectorAll('.chart-type-option').forEach(option => {
        if (option.value === selectedType) {
          option.selected = true;
        }
      });
      updateAllVisualizations();
    });
  }
}

/**
 * Configure les contrôles pour un conteneur de visualisation
 */
function setupVisualizationControls(container, index) {
  const chartTypeSelect = container.querySelector('.chart-type-option');
  const xAxisSelect = container.querySelector('.x-axis-select');
  const yAxisSelect = container.querySelector('.y-axis-select');
  const colorSelect = container.querySelector('.color-select');
  const updateButton = container.querySelector('.update-chart-btn');
  const chartDiv = container.querySelector('.chart-container');
  
  if (!chartDiv) return;
  
  // ID unique pour ce graphique
  const chartId = `chart-${index}`;
  chartDiv.id = chartId;
  
  // Créer un graphique par défaut au chargement
  if (chartTypeSelect && xAxisSelect) {
    createChart(
      chartId,
      chartTypeSelect.value,
      xAxisSelect.value,
      yAxisSelect ? yAxisSelect.value : null,
      colorSelect ? colorSelect.value : null
    );
  }
  
  // Configurer le bouton de mise à jour
  if (updateButton) {
    updateButton.addEventListener('click', function() {
      updateChart(
        chartId,
        chartTypeSelect.value,
        xAxisSelect.value,
        yAxisSelect ? yAxisSelect.value : null,
        colorSelect ? colorSelect.value : null
      );
    });
  }
}

/**
 * Crée un nouveau graphique
 */
function createChart(chartId, chartType, xVar, yVar, colorVar) {
  // Détruire le graphique précédent s'il existe
  if (activeCharts[chartId]) {
    activeCharts[chartId].destroy();
  }
  
  // Afficher un indicateur de chargement
  const chartContainer = document.getElementById(chartId);
  if (!chartContainer) return;
  
  chartContainer.innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Chargement de la visualisation...</p></div>';
  
  // Préparer les paramètres de la requête
  const params = {
    chart_type: chartType,
    x_var: xVar
  };
  
  if (yVar) params.y_var = yVar;
  if (colorVar && colorVar !== "Aucune") params.color_var = colorVar;
  
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
      showChartError(chartContainer, data.error);
      return;
    }
    
    // Nettoyer le conteneur
    chartContainer.innerHTML = '';
    
    // Créer le graphique avec Plotly
    Plotly.newPlot(chartId, data.data, data.layout, {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d']
    });
    
    // Stocker la référence au graphique
    activeCharts[chartId] = {
      destroy: function() {
        Plotly.purge(chartId);
      }
    };
  })
  .catch(error => {
    console.error('Erreur lors de la création du graphique:', error);
    showChartError(chartContainer, 'Erreur de communication avec le serveur');
  });
}

/**
 * Met à jour un graphique existant
 */
function updateChart(chartId, chartType, xVar, yVar, colorVar) {
  createChart(chartId, chartType, xVar, yVar, colorVar);
}

/**
 * Met à jour toutes les visualisations sur la page
 */
function updateAllVisualizations() {
  const visualizationContainers = document.querySelectorAll('.visualization-container');
  
  visualizationContainers.forEach((container, index) => {
    const chartId = `chart-${index}`;
    const chartTypeSelect = container.querySelector('.chart-type-option');
    const xAxisSelect = container.querySelector('.x-axis-select');
    const yAxisSelect = container.querySelector('.y-axis-select');
    const colorSelect = container.querySelector('.color-select');
    
    if (chartTypeSelect && xAxisSelect) {
      updateChart(
        chartId,
        chartTypeSelect.value,
        xAxisSelect.value,
        yAxisSelect ? yAxisSelect.value : null,
        colorSelect ? colorSelect.value : null
      );
    }
  });
}

/**
 * Affiche une erreur dans le conteneur du graphique
 */
function showChartError(container, message) {
  container.innerHTML = `
    <div class="alert alert-danger my-3">
      <i class="bi bi-exclamation-triangle-fill me-2"></i>
      <strong>Erreur de visualisation:</strong> ${message}
    </div>
  `;
}

/**
 * Exporte le graphique actuel au format PNG
 */
function exportChartAsPNG(chartId, filename) {
  if (!activeCharts[chartId]) return;
  
  const chartDiv = document.getElementById(chartId);
  
  Plotly.downloadImage(chartDiv, {
    format: 'png',
    filename: filename || 'orbit-chart',
    width: 1200,
    height: 800
  });
}

/**
 * Exporte le graphique actuel au format CSV
 */
function exportChartData(chartId, filename) {
  // Cette fonction nécessiterait un endpoint API supplémentaire
  // pour récupérer les données brutes du graphique
  console.log('Fonctionnalité d\'export CSV à implémenter');
}

/**
 * Configure le Dashboard interactif
 */
function setupDashboard(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  // Récupérer les statistiques générales pour le dashboard
  fetch('/api/dashboard-data', {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    }
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      container.innerHTML = `
        <div class="alert alert-danger">
          <i class="bi bi-exclamation-triangle-fill me-2"></i>
          ${data.error}
        </div>
      `;
      return;
    }
    
    // Créer les visualisations du dashboard
    createDashboardCharts(containerId, data);
  })
  .catch(error => {
    console.error('Erreur lors du chargement des données du dashboard:', error);
    container.innerHTML = `
      <div class="alert alert-danger">
        <i class="bi bi-exclamation-triangle-fill me-2"></i>
        Erreur de communication avec le serveur
      </div>
    `;
  });
}

/**
 * Crée les graphiques du dashboard
 */
function createDashboardCharts(containerId, data) {
  const container = document.getElementById(containerId);
  
  // Nettoyer le conteneur
  container.innerHTML = '';
  
  // Créer la structure du dashboard
  const dashboardHTML = `
    <div class="row">
      <div class="col-md-6 mb-4">
        <div class="card h-100">
          <div class="card-header d-flex justify-content-between align-items-center">
            <h5 class="mb-0">Distribution des variables numériques</h5>
            <button class="btn btn-sm btn-outline-primary export-chart-btn" data-chart="histogram-chart">
              <i class="bi bi-download"></i>
            </button>
          </div>
          <div class="card-body">
            <div id="histogram-chart" class="chart-container" style="height: 300px;"></div>
          </div>
        </div>
      </div>
      <div class="col-md-6 mb-4">
        <div class="card h-100">
          <div class="card-header d-flex justify-content-between align-items-center">
            <h5 class="mb-0">Matrice de corrélation</h5>
            <button class="btn btn-sm btn-outline-primary export-chart-btn" data-chart="correlation-chart">
              <i class="bi bi-download"></i>
            </button>
          </div>
          <div class="card-body">
            <div id="correlation-chart" class="chart-container" style="height: 300px;"></div>
          </div>
        </div>
      </div>
    </div>
    <div class="row">
      <div class="col-md-6 mb-4">
        <div class="card h-100">
          <div class="card-header d-flex justify-content-between align-items-center">
            <h5 class="mb-0">Valeurs manquantes</h5>
            <button class="btn btn-sm btn-outline-primary export-chart-btn" data-chart="missing-chart">
              <i class="bi bi-download"></i>
            </button>
          </div>
          <div class="card-body">
            <div id="missing-chart" class="chart-container" style="height: 300px;"></div>
          </div>
        </div>
      </div>
      <div class="col-md-6 mb-4">
        <div class="card h-100">
          <div class="card-header d-flex justify-content-between align-items-center">
            <h5 class="mb-0">Catégories principales</h5>
            <button class="btn btn-sm btn-outline-primary export-chart-btn" data-chart="categorical-chart">
              <i class="bi bi-download"></i>
            </button>
          </div>
          <div class="card-body">
            <div id="categorical-chart" class="chart-container" style="height: 300px;"></div>
          </div>
        </div>
      </div>
    </div>
  `;
  
  container.innerHTML = dashboardHTML;
  
  // Créer les graphiques individuels
  if (data.visualizations) {
    data.visualizations.forEach(viz => {
      if (document.getElementById(viz.id)) {
        Plotly.newPlot(viz.id, viz.data.data, viz.data.layout, {
          responsive: true,
          displayModeBar: false
        });
        
        // Stocker la référence
        activeCharts[viz.id] = {
          destroy: function() {
            Plotly.purge(viz.id);
          }
        };
      }
    });
  }
  
  // Configurer les boutons d'export
  document.querySelectorAll('.export-chart-btn').forEach(button => {
    button.addEventListener('click', function() {
      const chartId = this.getAttribute('data-chart');
      exportChartAsPNG(chartId, `orbit-${chartId}`);
    });
  });
}

// Initialiser les visualisations au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
  // Initialiser les visualisations si la page en contient
  if (document.querySelector('.visualization-container')) {
    initVisualizations();
  }
  
  // Initialiser le dashboard si présent
  if (document.getElementById('dashboard-container')) {
    setupDashboard('dashboard-container');
  }
});
