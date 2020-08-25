mkdir -p ~/.streamlit/

printf "\
[general]\n\
email = \"\"\
" > ~/.streamlit/credentials.toml

printf "\
[server]\n\
headless=true\n\
enableCORS=false\n\
enableXsrfProtection=false\n\
port=$PORT\n\
" > ~/.streamlit/config.toml
