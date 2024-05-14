import Chart from 'chart.js/auto'


var option_set = {
  // responsive: True,
  // maintainAspectRatio: True,
  scale: {
      ticks: {
          beginAtZero: true,
          min: 0
      }
  }
};

(async function() {
  const data = [
    { year: 2010, count: 10 },
    { year: 2011, count: 20 },
    { year: 2012, count: 15 },
    { year: 2013, count: 25 },
    { year: 2014, count: 22 },
    { year: 2015, count: 30 },
    { year: 2016, count: 28 },
  ];

  new Chart(
    document.getElementById('acquisitions'),
    {
      type: 'radar',
      data: {
        // labels: data.map(row => row.year),
        // datasets: [
        //   {
        //     label: 'Acquisitions by year',
        //     data: data.map(row => row.count)
        //   }
        // ]
        labels: ['danceability', 'energy', 'liveness', 'tempo', 'valence'],
        datasets: [
          {
            label: 'Artist A',
            backgroundColor: 'rgba(00, 255, 00, 0.1)',
            borderWidth: 2,
            data: [15, 35, 25, 20, 30]
          },
          {
            label: 'Artist B',
            backgroundColor: 'rgba(255, 00, 00, 0.1)',
            borderWidth: 2,
            data: [35, 10, 20, 25, 15]
          }
        ]
        
      },
      options: option_set
    }
  );
})();




