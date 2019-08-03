// Copyright 2018-19 PJ Engineering and Business Solutions Pty. Ltd. All rights reserved.

package dataframe

import (
	"errors"
)

// ErrNoRows signifies that the Series, Dataframe or import data
// contains no rows of data.
var ErrNoRows = errors.New("contains no rows")

// ErrIndeterminate indicates that the result of a calculation is indeterminate.
var ErrIndeterminate = errors.New("calculation result: indeterminate")
