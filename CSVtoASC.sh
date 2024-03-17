mkdir ../ASC
for i in *.csv; do
    tmp=$( basename "$i" .csv );
    sed 's/,/\n/3;P;D' $i > /tmp/$tmp.asc
    sed 's/,/ /g' /tmp/$tmp.asc > ../ASC/$tmp.asc
    rm /tmp/$tmp.asc
done;
