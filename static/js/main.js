/**
 * FAUCON - Main JavaScript
 * Fonctionnalités communes à toutes les pages
 */

document.addEventListener('DOMContentLoaded', function() {
  // Animation des cartes
  const cards = document.querySelectorAll('.card');
  cards.forEach(card => {
    if (card.classList.contains('no-hover-effect')) return;
    card.addEventListener('mouseenter', function() {
      card.style.boxShadow = '0 8px 16px rgba(0, 0, 0, 0.1)';
    });
    card.addEventListener('mouseleave', function() {
      card.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
    });
  });

  // Gestion des messages flash
  const flashMessages = document.querySelectorAll('.alert-dismissible');
  flashMessages.forEach(message => {
    // Auto-fermeture après 5 secondes
    setTimeout(() => {
      const closeButton = message.querySelector('.btn-close');
      if (closeButton) {
        closeButton.click();
      }
    }, 5000);
  });

  // Activer les tooltips Bootstrap
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Gestion du thème sombre/clair si présent
  const themeToggler = document.getElementById('theme-toggle');
  if (themeToggler) {
    themeToggler.addEventListener('click', function() {
      document.body.classList.toggle('dark-theme');
      
      // Sauvegarder la préférence
      const isDarkMode = document.body.classList.contains('dark-theme');
      localStorage.setItem('darkMode', isDarkMode.toString());
      
      // Mettre à jour l'icône
      const icon = this.querySelector('i');
      if (isDarkMode) {
        icon.classList.remove('bi-moon');
        icon.classList.add('bi-sun');
      } else {
        icon.classList.remove('bi-sun');
        icon.classList.add('bi-moon');
      }
    });
    
    // Charger la préférence au chargement
    const storedDarkMode = localStorage.getItem('darkMode');
    if (storedDarkMode === 'true') {
      document.body.classList.add('dark-theme');
      const icon = themeToggler.querySelector('i');
      if (icon) {
        icon.classList.remove('bi-moon');
        icon.classList.add('bi-sun');
      }
    }
  }

  // Gestion des zones de glisser-déposer
  const dropZones = document.querySelectorAll('.drag-drop-zone');
  if (dropZones.length > 0) {
    setupDragDropZones();
  }

  // Formatage des tableaux de données
  const dataTables = document.querySelectorAll('.data-table');
  dataTables.forEach(table => {
    // Ajouter des classes pour rendre les tableaux plus lisibles
    table.classList.add('table', 'table-striped', 'table-hover');
    
    // Ajouter des en-têtes cliquables pour le tri si la classe est présente
    if (table.classList.contains('sortable')) {
      setupSortableTable(table);
    }
  });

  // Gestion du menu mobile
  const mobileMenuToggle = document.querySelector('.navbar-toggler');
  if (mobileMenuToggle) {
    mobileMenuToggle.addEventListener('click', function() {
      document.body.classList.toggle('mobile-menu-open');
    });
  }
});

/**
 * Configure les zones de glisser-déposer pour le téléchargement de fichiers
 */
function setupDragDropZones() {
  const dropZones = document.querySelectorAll('.drag-drop-zone');
  
  dropZones.forEach(zone => {
    const fileInput = zone.querySelector('input[type="file"]');
    if (!fileInput) return;
    
    // Empêcher le navigateur d'ouvrir les fichiers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      zone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }
    
    // Gérer les effets visuels
    ['dragenter', 'dragover'].forEach(eventName => {
      zone.addEventListener(eventName, () => {
        zone.classList.add('active');
      });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
      zone.addEventListener(eventName, () => {
        zone.classList.remove('active');
      });
    });
    
    // Gérer le dépôt
    zone.addEventListener('drop', function(e) {
      fileInput.files = e.dataTransfer.files;
      
      // Déclencher l'événement change pour mettre à jour l'interface
      const event = new Event('change', { bubbles: true });
      fileInput.dispatchEvent(event);
      
      // Afficher le nom du fichier si un élément est désigné pour cela
      const fileNameDisplay = document.getElementById('selectedFileName');
      if (fileNameDisplay && fileInput.files.length > 0) {
        fileNameDisplay.textContent = `Fichier sélectionné: ${fileInput.files[0].name}`;
      }
      
      // Activer le bouton de téléchargement s'il existe
      const uploadButton = document.getElementById('uploadButton');
      if (uploadButton) {
        uploadButton.disabled = false;
      }
    });
  });
}

/**
 * Configure les tableaux pour être triables
 */
function setupSortableTable(table) {
  const headers = table.querySelectorAll('th');
  
  headers.forEach(header => {
    header.style.cursor = 'pointer';
    header.addEventListener('click', function() {
      const index = Array.from(header.parentElement.children).indexOf(header);
      sortTable(table, index, header.getAttribute('data-sort-direction') === 'asc');
      
      // Mettre à jour la direction pour le prochain clic
      headers.forEach(h => h.removeAttribute('data-sort-direction'));
      const newDirection = header.getAttribute('data-sort-direction') === 'asc' ? 'desc' : 'asc';
      header.setAttribute('data-sort-direction', newDirection);
      
      // Mise à jour des indicateurs visuels
      headers.forEach(h => {
        h.querySelector('.sort-indicator')?.remove();
      });
      
      const indicator = document.createElement('span');
      indicator.className = 'sort-indicator ms-1';
      indicator.innerHTML = newDirection === 'asc' ? '↓' : '↑';
      header.appendChild(indicator);
    });
  });
}

/**
 * Trie un tableau HTML
 */
function sortTable(table, column, ascending) {
  const rows = Array.from(table.querySelectorAll('tbody tr'));
  const body = table.querySelector('tbody');
  
  // Trier les lignes
  const sortedRows = rows.sort((a, b) => {
    const cellA = a.children[column].textContent.trim();
    const cellB = b.children[column].textContent.trim();
    
    // Essayer de convertir en nombre si possible
    const numA = parseFloat(cellA);
    const numB = parseFloat(cellB);
    
    if (!isNaN(numA) && !isNaN(numB)) {
      return ascending ? numA - numB : numB - numA;
    }
    
    // Sinon, comparer en tant que chaînes
    return ascending 
      ? cellA.localeCompare(cellB) 
      : cellB.localeCompare(cellA);
  });
  
  // Vider le tbody
  while (body.firstChild) {
    body.removeChild(body.firstChild);
  }
  
  // Ajouter les lignes triées
  sortedRows.forEach(row => body.appendChild(row));
}
