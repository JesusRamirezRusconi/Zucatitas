import streamlit as st

def main():
    st.title("Aplicación de prueba para Streamlit")
    st.write("Esta es una aplicación de prueba para verificar que Streamlit funciona correctamente.")
    
    st.header("Prueba de interactividad")
    name = st.text_input("¿Cómo te llamas?")
    if name:
        st.write(f"Hola, {name}!")

if __name__ == "__main__":
    main()

