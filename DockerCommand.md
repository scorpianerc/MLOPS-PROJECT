docker-compose up -d

# Stop dan hapus semua container
docker-compose down

# Stop tanpa hapus (bisa restart cepat)
docker-compose stop

# Start ulang setelah stop
docker-compose start

# Restart container tertentu
docker-compose restart streamlit

# Lihat status
docker-compose ps

# Lihat logs
docker-compose logs -f

# Hapus semua + volumes (HATI-HATI: hapus data!)
docker-compose down -v