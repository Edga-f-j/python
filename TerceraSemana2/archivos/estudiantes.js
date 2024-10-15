let estudiantes = [
    {
        "nombre": "pepito",
        "edad": 24,
        "matematicas": 92,
        "ciencias": 84,
        "ingles": 65,
        "sociales": 34
    },
    {
        "nombre": "juan",
        "edad": 22,
        "matematicas": 72,
        "ciencias": 54,
        "ingles": 35,
        "sociales": 64
    },
    {
        "nombre": "luca",
        "edad": 17,
        "matematicas": 52,
        "ciencias": 98,
        "ingles": 67,
        "sociales": 48
    },
    {
        "nombre": "kaiser",
        "edad": 19,
        "matematicas": 50,
        "ciencias": 64,
        "ingles": 77,
        "sociales": 86
    },
    {
        "nombre":"ismael",
        "edad":16,
        "matematicas": 78,
        "ciencias": 89,
        "ingles": 89,
        "sociales": 67
    },

    {
        "nombre":"gasparin",
        "edad":26,
        "matematicas": 80,
        "ciencias": 80,
        "ingles": 89,
        "sociales": 98
    },

    {
        "nombre":"arturo",
        "edad":24,
        "matematicas": 34,
        "ciencias": 36,
        "ingles": 89,
        "sociales": 77
    }
]


const tabla = document.querySelector("#estudiantes tbody");

estudiantes.forEach(estudiante =>{
    let fila = document.createElement('tr');


    fila.innerHTML=`
    <td>${estudiante.nombre}</td>
    <td>${estudiante.edad}</td>
    <td>${estudiante.matematicas}</td>
    <td>${estudiante.ciencias}</td>
    <td>${estudiante.ingles}</td>
    <td>${estudiante.sociales}</td>
    `;

    document.querySelector("#estudiantes tbody").appendChild(fila);
    //tabla.appendChild(fila);

})









function mostrarEstudiantes(filtrados) {
    const tbody = document.querySelector("#estudiantes tbody");
    tbody.innerHTML = '';  // Limpiamos el contenido actual de la tabla
  
    // Iteramos sobre los estudiantes filtrados
    for(let i = 0; i < filtrados.length; i++) {
      let fila = document.createElement('tr');
      let estudiante = filtrados[i];
  
      fila.innerHTML = `
        <td>${estudiante.nombre}</td>
        <td>${estudiante.edad}</td>
        <td>${estudiante.matematicas}</td>
        <td>${estudiante.ciencias}</td>
        <td>${estudiante.ingles}</td>
        <td>${estudiante.sociales}</td>
      `;
  
      tbody.appendChild(fila);  // Añadimos la fila a la tabla
    }
  }





  
  function filtrarEstudiantes() {
    // Obtenemos los valores de las materias y del nombre
    const nombreSeleccionado = document.querySelector('#filtroNombre').value;
    const notaMatematicas = document.querySelector('#notaMatematicas').value;
    const notaCiencias = document.querySelector('#notaCiencias').value;
    const notaIngles = document.querySelector('#notaIngles').value;
    const notaSociales = document.querySelector('#notaSociales').value;
  
    // Filtramos los estudiantes según los criterios
    const estudiantesFiltrados = estudiantes.filter(estudiante => {
      return (
        (nombreSeleccionado === "" || estudiante.nombre === nombreSeleccionado) && // Filtrar por nombre
        (notaMatematicas === "" || estudiante.matematicas >= notaMatematicas) &&
        (notaCiencias === "" || estudiante.ciencias >= notaCiencias) &&
        (notaIngles === "" || estudiante.ingles >= notaIngles) &&
        (notaSociales === "" || estudiante.sociales >= notaSociales)
      );
    });
  
    // Mostramos los estudiantes filtrados
    mostrarEstudiantes(estudiantesFiltrados);
  }
  
  // Mostramos todos los estudiantes al inicio
  mostrarEstudiantes(estudiantes);
  
  
  // Llamamos a esta función al cargar la página para mostrar todos los estudiantes inicialmente




// for(let i = 0; i <=estudiantes.length-1; i++){
//     let fila = document.createElement('tr');
//     let estudiante = estudiantes[i]

//     fila.innerHTML=`
//     <td>${estudiantes[i].nombre}</td>
//     <td>${estudiante.edad}</td>
//     <td>${estudiante.matematicas}</td>
//     <td>${estudiante.ciencias}</td>
//     <td>${estudiante.ingles}</td>
//     <td>${estudiante.sociales}</td>
//     `;


//     document.querySelector("#estudiantes tbody").appendChild(fila);

// }
//appendchild
//docment.createElement
//document.querySelector
//foreach
//fila.innerHTML