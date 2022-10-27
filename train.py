from process import num_epochs, train_dataset, train_batch_size, eval_dataset, custom_optimizer

model.train(num_epochs = num_epochs, 
            train_dataset = train_dataset, 
            train_batch_size=train_batch_size, 
            eval_dataset=eval_dataset, 
            optimizer=custom_optimizer, 
            save_interval_epochs=1, 
            log_interval_steps=2, 
            save_dir='output/T001', 
            pretrain_weights='COCO', 
            metric=None, 
            early_stop=True, 
            early_stop_patience=5, 
            use_vdl=True#,
            #pretrain_weights = None,
            #resume_checkpoint = "output/T008_101_vdMCpie3*lr/epoch_38_78.376"
)