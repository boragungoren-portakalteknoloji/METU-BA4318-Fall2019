mkdir wget-log
mkdir cubic-out
mkdir lead-out

echo "178.157.14.137 IP adresinde TCP CUBIC yüklü sunucudan indirme başlıyor."
echo "Eşzamanlı olarak 178.157.14.138 IP adresinde TCP LEAD yüklü sunucudan indirme başlıyor."

wget -P cubic-out --timestamping -o ./wget-log/interview-cubic.out --report-speed=bits --retry-connrefused --wait=10 --random-wait --no-proxy --no-cache --no-dns-cache http://178.157.14.137/media/interview.mp4 &
wget -P lead-out --timestamping -o ./wget-log/interview-lead.out --report-speed=bits --retry-connrefused --wait=10 --random-wait --no-proxy --no-cache --no-dns-cache http://178.157.14.138/media/interview.mp4 &

wget -P cubic-out --timestamping -o ./wget-log/astro-cubic.out --report-speed=bits --retry-connrefused --wait=10 --random-wait --no-proxy --no-cache --no-dns-cache http://178.157.14.137/media/nasaAstronauts.mp4 &
wget -P lead-out --timestamping -o ./wget-log/astro-lead.out --report-speed=bits --retry-connrefused --wait=10 --random-wait --no-proxy --no-cache --no-dns-cache http://178.157.14.138/media/nasaAstronauts.mp4 &

wget -P cubic-out --timestamping -o ./wget-log/guppy-cubic.out --report-speed=bits --retry-connrefused --wait=10 --random-wait --no-proxy --no-cache --no-dns-cache http://178.157.14.137/media/superGuppy.mp4 &
wget -P lead-out --timestamping -o ./wget-log/guppy-lead.out --report-speed=bits --retry-connrefused --wait=10 --random-wait --no-proxy --no-cache --no-dns-cache http://178.157.14.138/media/superGuppy.mp4 &

wait

echo "İndirilen dosyalar siliniyor. Kayıtlar wget-log dizini içindedir."
rm -rf index.html
rm -rf *.mp4
rm -rf cubic-out/*.mp4
rm -rf lead-out/*.mp4
