# fin click ML
library(PAMpal)
db <- '../R_Projects/Data/CHW_Fin/Databases/'
bin <- '../R_Projects/Data/CHW_Fin/Binaries/'
pps <- PAMpalSettings(db=db, binaries = bin, sr_hz=200, filterfrom_khz=0, filterto_khz=NULL, winLen_sec=1)
dataDb <- processPgDetections(pps, mode='db', id='FinClickDB')
dataRec <- processPgDetections(pps, mode='recording', id='FinClickRecording')

saveRDS(dataDb, file='../FinClickCV/dbStudy.rds')
saveRDS(dataRec, file='../FinClickCV/recStudy.rds')
# look at some
hm <- getBinaryData(dataDb[1:5], getClickData(dataDb[1:5])$UID)

wl <- 256
clip <- wigPreproc(hm[[5]], srTo=200, c=1, wl=wl)
par(mfrow=c(1,2))

plot(clip, type='l')
wig <- PAMmisc::wignerTransform(clip, sr=200, plot=T)

# write wiggys
det1only <- filter(dataRec, detectorName == 'Fin_Whale_20_Hz_Detector_1')
for(e in seq_along(events(det1only))) {
    settings(det1only[[e]])$sr <- 200
}
det1db <- filter(dataDb, detectorName == 'Fin_Whale_20_Hz_Detector_1')
dbDf <- getClickData(det1db)
dbDf$drift <- gsub('CCES_2018_(DRIFT[0-9]{1,2})_.*', '\\1', basename(dbDf$db))
det1df <- processAllWig(det1only, wl=256,sr=200, dir='../FinClickCV/data/Detector1', dataset = 'finclick')
det1df <- det1df[det1df$BinaryFile != 'Click_Detector_Fin_Whale_20_Hz_Detector_Clicks_20180927_121844(1).pgdf', ]
dbDf$label <- 1

hm <- left_join(det1df, distinct(dbDf[c('drift', 'UID', 'label')]), by=c('UID'='UID', 'drift'='drift'))
hm$label[is.na(hm$label)] <- 0
saveRDS(hm, file='../FinClickCV/FinClickDF_All.rds')
# she has duplicate clicks in differnet events so dbDf is all fucked with 5 dups
dups <- duplicated(select(dbDf, UID, drift))
sum(dups)
View(dbDf[dbDf$UID %in% dbDf$UID[dups],])
View(det1df[det1df$UID == '2781000015',])
