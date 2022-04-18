#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import io, transform
from sklearn.model_selection import train_test_split
from typing import Union

class loadAnnotations:
    def __init__(self):
        self.annotations = pd.DataFrame()
        self.reviewed = pd.DataFrame()
        self.col_path = None
    
    def load_all(self, csv_paths, **kwds):
        csv_paths = glob(csv_paths)
        for csv_path in csv_paths:
            self.load(csv_path=csv_path, append=True, **kwds)
    
    def load(self, csv_path, path2dir=None, col_path="image_id", 
             keep_these_cols=False, append=True, col_label="label", 
             shuffle_rows=True, reset_index=True, random_state=1234):
        """Read/append annotation file(s)

        Parameters
        ----------
        csv_path : str
            path to an annotation file in CSV format
        path2dir : str, optional
            update col_path by adding path2dir/col_path, by default None
        col_path : str, optional
            column that contains image paths, by default "image_id"
        keep_these_cols : bool, optional
            only keep these columns, if False (default), all columns will be kept
        append : bool, optional
            append a newly read csv file to self.annotations, by default True
        col_label : str
            Name of the column that contains labels
        shuffle_rows : bool
            Shuffle rows after reading annotations
        """
        if isinstance(csv_path, str):
            print(f"* reading: {csv_path}")
            annots_rd = pd.read_csv(csv_path)
        else:
            print(f"* reading dataframe")
            annots_rd = csv_path.copy()
        self.col_label = col_label
        print(f"* #rows: {len(annots_rd)}")
        print(f"* label column name: {self.col_label} (you can change this later by .set_col_label(new_label) )")
        if shuffle_rows:
            annots_rd = annots_rd.sample(frac=1, random_state=random_state)
            print("* shuffle rows: Yes")

        if keep_these_cols:
            annots_rd = annots_rd[keep_these_cols]

        if self.col_path == None:
            self.col_path = col_path
        elif self.col_path != col_path:
            print(f"[WARNING] previously, the col_path was set to {self.col_path}. Column '{col_path}' will be renamed.")
            annots_rd.rename(columns={col_path: self.col_path}, inplace=True)
        
        if path2dir:
            print(f"* update paths in '{self.col_path}' column by inserting '{path2dir}'")
            annots_rd[self.col_path] = os.path.abspath(path2dir) + os.path.sep + annots_rd[self.col_path]
        
        if (len(self.annotations) == 0) or (append==False):
            self.annotations = annots_rd.copy()
        else:
            self.annotations = pd.concat([self.annotations, annots_rd.copy()], ignore_index=True)
        
        self.annotations.drop_duplicates(subset=[self.col_path], inplace=True)
        if reset_index:
            self.annotations.reset_index(drop=True, inplace=True)
        print()
        print(self)
    
    def set_col_label(self, new_label: str = "label"):
        """Set the name of the column that contains labels

        Parameters
        ----------
        new_label : str, optional
            Name of the column that contains labels, by default "label"
        """
        self.col_label = new_label
   
    def show_image(self, 
                   indx: int,
                   cmap="viridis"
                   ):
        """Show an image by its index (i.e., iloc in pandas)

        Parameters
        ----------
        indx : int
            Index of the image to be plotted 
        """
        if (self.col_path == None) or (len(self.annotations) == 0):
            print(f"[ERROR] length: {len(self.annotations)}")
            return

        plt.imshow(io.imread(self.annotations.iloc[indx][self.col_path]), cmap=cmap)
        plt.title(self.annotations.iloc[indx][self.col_label])
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.001)
        plt.show()
    
    def adjust_labels(self,
                      shiftby: int=-1):
        """Shift labels by the specified value (shiftby)

        Parameters
        ----------
        shiftby : int, optional
            shift values of self.col_label by shiftby, i.e., self.annotations[self.col_label] + shiftby, by default -1
        """
        print(20*"-")
        print("[INFO] value counts before shift:")
        print(self.annotations[self.col_label].value_counts())

        self.annotations[self.col_label] += shiftby

        print(20*"-")
        print("[INFO] value counts after shift:")
        print(self.annotations[self.col_label].value_counts())
        print(20*"-")
    
    def review_labels(self, 
                      tar_label: Union[None, int]=None, 
                      start_indx: int=1, 
                      chunks: int=8*6, 
                      num_cols: int=8,
                      figsize: Union[list, tuple]=(8*3, 8*2),
                      exclude_df=None,
                      include_df=None,
                      deduplicate_col: str="image_id"):
        """Review/edit labels

        Parameters
        ----------
        tar_label : Union[None, int], optional
        start_indx : int, optional
        chunks : int, optional
        num_cols : int, optional
        figsize : Union[list, tuple], optional
        """

        if tar_label is not None:
            annot2review = self.annotations[self.annotations[self.col_label] == tar_label]
        else:
            annot2review = self.annotations
        
        annot2review.drop_duplicates(inplace=True)
        
        indx = start_indx - 1
        while indx < len(annot2review):
            plt.figure(figsize=figsize)
            print("\n" + 30*"*")
            print(f"[INFO] review {indx+1}-{indx+chunks}, total: {len(annot2review)}")
            print(30*"*")
            
            counter = 1
            iter_ids = []
            while (counter <= chunks) and (indx < len(annot2review)):
                # Skip the image if it is in exclude_df
                if exclude_df is not None:
                    if annot2review.iloc[indx]["image_path"] in exclude_df["image_path"].to_list():
                        indx += 1
                        continue
                
                # Skip the image if it is NOT in include_df
                if include_df is not None:
                    if annot2review.iloc[indx]["image_path"] not in exclude_df["image_path"].to_list():
                        indx += 1
                        continue

                # The first term is just a ceiling division, equivalent to:
                # from math import ceil
                # int(ceil(chunks / num_cols))
                plt.subplot(-(-chunks // num_cols), num_cols, counter) 
                plt.imshow(io.imread(annot2review.iloc[indx][self.col_path]))
                plt.xticks([]) 
                plt.yticks([])
                plt.title(f"{annot2review.iloc[indx][self.col_label]} | id: {annot2review.iloc[indx].name}") 
                iter_ids.append(annot2review.iloc[indx].name)
                # Add to reviewed
                self.reviewed = self.reviewed.append(annot2review.iloc[indx])
                try:
                    self.reviewed.drop_duplicates(subset=[deduplicate_col])
                except Exception:
                    pass
                counter += 1
                indx += 1
            plt.show()

            print(f"list of IDs: {iter_ids}")
            user_input_ids = input("Enter 'ids', comma separated (or press enter to continue)  :  ")

            while user_input_ids.strip().lower() not in ["", "exit", "end", "stop"]:
                list_input_ids = user_input_ids.split(",")
                input_label = int(input("Enter label  :  "))

                for one_input_id in list_input_ids:
                    input_id = int(one_input_id)
                    # Change both annotations and reviewed
                    self.annotations.loc[input_id, self.col_label] = input_label
                    self.reviewed.loc[input_id, self.col_label] = input_label
                    print(f"{input_id} ---> new label: {input_label}")

                user_input_ids = input("Enter 'ids', comma separated (or press enter to continue)  :  ")
            
            if user_input_ids.lower() in ["exit", "end", "stop"]:
                break

        print("[INFO] Exit...")
    
    def show_image_labels(self, tar_label=1, num_sample=10):
        """Show sample images for the specified label

        Parameters
        ----------
        tar_label : int, optional
            target label to be used in plotting, by default 1
        num_sample : int, optional
            number of samples to plot, by default 10
        """
        if (self.col_path == None) or (len(self.annotations) == 0):
            print(f"[ERROR] length: {len(self.annotations)}")
            return

        annot2plot = self.annotations[self.annotations[self.col_label] == tar_label]
        
        if num_sample == None:
            num_sample = len(annot2plot)

        plt.figure(figsize=(8, num_sample))
        for indx in range(num_sample):
            plt.subplot(int(num_sample/2.), 3, indx+1)
            plt.imshow(io.imread(annot2plot.iloc[indx][self.col_path]))
            plt.xticks([])
            plt.yticks([])
            plt.title(annot2plot.iloc[indx][self.col_label])
        plt.show()

    def split_annotations(self, stratify_colname='label',
                          frac_train=0.70, frac_val=0.15, frac_test=0.15,
                          random_state=1364):
        """Split pandas dataframe into three subsets.

        CREDIT: https://stackoverflow.com/a/60804119 (with minor changes)

        Following fractional ratios provided by the user, where each subset is
        stratified by the values in a specific column (that is, each subset has
        the same relative frequency of the values in the column). It performs this
        splitting by running train_test_split() twice.
    
        Parameters
        ----------
        stratify_colname : str
            The name of the column that will be used for stratification. 
        frac_train : float
        frac_val   : float
        frac_test  : float
            The ratios with which the dataframe will be split into train, val, and
            test data. The values should be expressed as float fractions and should
            sum to 1.0.
        random_state : int, None, or RandomStateInstance
            Value to be passed to train_test_split().
    
        Returns
        -------
        df_train, df_val, df_test :
            Dataframes containing the three splits.
        """
    
        if abs(frac_train + frac_val + frac_test - 1.0) > 1e-4:
            raise ValueError(f'fractions {frac_train}, {frac_val}, {frac_test} do not add up to 1.0.' 
                             f'Their sum: {frac_train+frac_val+frac_test}')
    
        if stratify_colname not in self.annotations.columns:
            raise ValueError(f'{stratify_colname} is not a column in the dataframe')
    
        X = self.annotations # Contains all columns.
        y = X[[stratify_colname]] # Dataframe of just the column on which to stratify.
    
        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                              y,
                                                              stratify=y,
                                                              test_size=(1.0 - frac_train),
                                                              random_state=random_state)
    
        if abs(frac_test) < 1e-3:
            df_val = df_temp
            df_test = None
            assert len(self.annotations) == len(df_train) + len(df_val)
        else:
            # Split the temp dataframe into val and test dataframes.
            relative_frac_test = frac_test / (frac_val + frac_test)
            df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                            y_temp,
                                                            stratify=y_temp,
                                                            test_size=relative_frac_test,
                                                            random_state=random_state)
            assert len(self.annotations) == len(df_train) + len(df_val) + len(df_test)

        self.train = df_train
        self.val = df_val
        self.test = df_test
        print("---------------------")
        print("* Split dataset into:")
        print(f"    Train: {len(self.train)}")
        print(f"    Valid: {len(self.val)}")
        print(f"    Test : {len(self.test) if self.test is not None else 0}")
        print("---------------------")

    def sample_labels(self, tar_label, num_samples, random_state=12345):

        if (self.col_path == None) or (len(self.annotations) == 0):
            print(f"[ERROR] length: {len(self.annotations)}")
            return
        all_annots = self.annotations.copy() 
        tar_rows = all_annots[all_annots[self.col_label] == tar_label]
        tar_samples = tar_rows.sample(num_samples, random_state=random_state)
        new_annots = all_annots[(all_annots[self.col_label] != tar_label) | (all_annots.index.isin(tar_samples.index))]
        self.annotations = new_annots
        
    def __str__(self):
        print(f"------------------------")
        print(f"* Number of annotations: {len(self.annotations)}\n")
        if len(self.annotations) > 0:
            print(f"* First few rows:")
            print(self.annotations.head())
            print("...\n")
            print(f"* Value counts (column: {self.col_label}):")
            print(self.annotations[self.col_label].value_counts())
        print(f"------------------------")
        return ""
